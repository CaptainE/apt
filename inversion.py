import copy

import lpips
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.nn import functional as F
from metrics import metric_utils
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

criterion = torch.nn.CrossEntropyLoss()

def crossentropy_loss(logits, new_class):
    return criterion(logits, new_class)

class ClassProbabilitiesPredictor():
    def __init__(self, generator, device) -> None:
        self.device = device
        self.must_get_class_prob = self.determine_must_get_class_probabilities(generator)
        if self.must_get_class_prob:
            self.nontrainable_classifier_for_imgnet_prediction = timm.create_model('deit_base_distilled_patch16_224', pretrained=True).eval().to(device)

    def determine_must_get_class_probabilities(self, generator):
        if not generator.c_dim:
            self.c_dim = None
            return True
        else:
            self.c_dim = generator.c_dim
            return False

    def get_class_probabilities(self, w_avg_samples, target_img):
        # get class probas by classifier

        cls_target = F.interpolate((target_img.to(self.device).to(torch.float32) / 127.5 - 1)[None], 224)
        logits = self.nontrainable_classifier_for_imgnet_prediction(cls_target).softmax(1)
        classes = torch.multinomial(logits, w_avg_samples, replacement=True).squeeze()
        print(f'Main class: {logits.argmax(1).item()}, confidence: {logits.max().item():.4f}')
        c_samples = np.zeros([w_avg_samples, self.c_dim], dtype=np.float32)
        for i, c in enumerate(classes):
            c_samples[i, c] = 1
        c_samples = torch.from_numpy(c_samples).to(self.device)
        
        return c_samples

    def prepare_target_images(self, target):
        target_images = target.unsqueeze(0).to(self.device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        return target_images

class PTILossModule():
    def __init__(self, classprobpredictor:ClassProbabilitiesPredictor) -> None:
        self.has_no_c_dim = classprobpredictor.must_get_class_prob
        self.generator_c_dim = classprobpredictor.c_dim
        self.device = classprobpredictor.device
        self.c_sampling_function = self.determine_c_sampling_function()

    def determine_c_sampling_function(self):
        if self.has_no_c_dim:
            return lambda x: None
        else:
            return lambda x: (F.one_hot(torch.randint(self.generator_c_dim, (x,)), self.generator_c_dim)).to(self.device)

    def get_morphed_w_code(self, new_w_code, fixed_w, regularizer_alpha=30):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = regularizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        return result_w

    def space_regularizer_loss(self, 
        G_original,
        G_pti,
        w_batch,
        vgg16,
        z_samples, 
        c_samples,
        lpips_lambda=10,
    ):

        w_samples = G_original.mapping(z_samples, c_samples, truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = G_pti.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = G_original.synthesis(w_code, noise_mode='none', force_fp32=True)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            if new_img.shape[-1] > 256:
                new_img = F.interpolate(new_img, size=(256, 256), mode='area')
                old_img = F.interpolate(old_img, size=(256, 256), mode='area')

            new_feat = vgg16(new_img, resize_images=False, return_lpips=True)
            old_feat = vgg16(old_img, resize_images=False, return_lpips=True)
            lpips_loss = lpips_lambda * (old_feat - new_feat).square().sum()

        return lpips_loss / len(territory_indicator_ws)

class LossModule():
    def __init__(self, classprobpredictor:ClassProbabilitiesPredictor, device) -> None:
        self.device= device
        self.percepual_network_loss = lpips.LPIPS(net='vgg').to(device) 
        self.vgg16 = self.load_feature_detector()
        self.pti_loss = PTILossModule(classprobpredictor)
        self.l2_criterion = torch.nn.MSELoss(reduction='mean')

    def load_feature_detector(self):
    # Load VGG16 feature detector.
        vgg16_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
        vgg16 = metric_utils.get_feature_detector(vgg16_url, device=self.device)
        return vgg16
        
    def classifier_crossentropy_loss_on_generated_class_to_predict(self, classifier_logits, class_to_predict ):
        predicting_class = torch.tensor(class_to_predict,device=self.device, dtype=torch.int64, requires_grad=False)
        targets =  predicting_class.view(-1)
        return crossentropy_loss(classifier_logits, targets)

class LatentOptimizationModule():
    def __init__(self, classprobpredictor:ClassProbabilitiesPredictor, generator, lossmodule:LossModule) -> None:
        self.classprobpredictor = classprobpredictor
        self.must_get_class_prob = classprobpredictor.must_get_class_prob
        self.generator_c_dim = classprobpredictor.c_dim
        self.device = classprobpredictor.device
        self.c_sampling_function = self.determine_c_sampling_function()
        self.generator = generator
        self.lossmodule = lossmodule
        
        self.initial_learning_rate = 0.05
        self.lr_rampdown_length = 0.25
        self.lr_rampup_length = 0.05

    def make_c_samples(self, w_avg_samples, target, target_class):
        c_samples = torch.tensor(np.zeros([1, self.generator_c_dim], dtype=np.float32)).to(self.device)
        c_samples[:,target_class]=1
        c_samples = c_samples.repeat(w_avg_samples,1)
        return c_samples

    def determine_c_sampling_function(self):
        if not self.must_get_class_prob:
            return lambda w_avg_samples, target, target_class: self.make_c_samples(w_avg_samples, target, target_class)
        else:
            return lambda w_avg_samples, target, target_class: self.classprobpredictor.get_class_probabilities(w_avg_samples, target)

    def lr_scheduler_step(self, step, num_steps, optimizer):
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        lr = self.initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def project(self, 
        target_class: torch.Tensor, # [1,1000]
        target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        num_steps = 1000,
        w_avg_samples = 10000,
        verbose = False,
        noise_mode="const",
    ):
        assert target.squeeze(0).shape == (self.generator.img_channels, self.generator.img_resolution, self.generator.img_resolution)

        # Compute w stats.
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        with torch.no_grad():
            z_samples = torch.from_numpy(np.random.RandomState(123).randn(w_avg_samples, self.generator.z_dim)).to(self.device)

            c_samples = self.c_sampling_function(w_avg_samples, target, target_class)

            w_samples = self.generator.mapping(z_samples, c_samples)  # [N, L, C]

            # get empirical w_avg
            w_avg = w_samples.mean(0)      # [L, C]

        # Features for target image.
        target_images = self.classprobpredictor.prepare_target_images(target)
        target_features = self.lossmodule.vgg16(target_images, resize_images=False, return_lpips=True)

        # initalize optimizer

        w_opt = w_avg.detach().clone().unsqueeze(0)[:,0:1,:]  #[1, 1, C]
        w_opt.requires_grad = True
        #w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True) # pylint: disable=not-callable
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=self.initial_learning_rate)

        # run optimization loop
        all_images = []
        for step in range(num_steps):
            # Learning rate schedule.
            self.lr_scheduler_step(step, num_steps, optimizer)

            # Synth images from opt_w.
            synth_images = self.generator.synthesis(w_opt[0].repeat(1,self.generator.num_ws,1), noise_mode=noise_mode)

            # track images
            synth_images = (synth_images + 1) * (255/2)

            synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            all_images.append(synth_images_np)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = self.lossmodule.vgg16(synth_images, resize_images=False, return_lpips=True)
            lpips_loss = (target_features - synth_features).square().sum()
            

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss = lpips_loss
            loss.backward()
            optimizer.step()
            msg  = f'[ step {step+1:>4d}/{num_steps}] '
            msg += f'[ loss: {float(loss):<5.2f}] '
            if verbose: print(msg)

        return all_images, w_opt.detach()[0]

class PTIOptimization():
    def __init__(self, device, generator, discriminator, classifier):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier

        self.CPP = ClassProbabilitiesPredictor(self.generator, device)
        self.LM = LossModule(self.CPP, device)
        self.LO = LatentOptimizationModule(self.CPP,self.generator,self.LM)


    def pivotal_tuning(self,  w_pivot, target_class, target, num_steps, learning_rate, noise_mode, verbose):

        G_pti = copy.deepcopy(self.generator).train().requires_grad_(True).to(self.device)
        w_pivot.requires_grad_(False)

        # Features for target image.
        target_images = self.CPP.prepare_target_images(target)
        target_features = self.LM.vgg16(target_images, resize_images=False, return_lpips=True)

        # initalize optimizer
        optimizer = torch.optim.Adam(G_pti.parameters(), lr=learning_rate)
        
        # run optimization loop
        all_images = []
        for step in range(num_steps):
            # Synth images from opt_w.
            synth_images = G_pti.synthesis(w_pivot[0].repeat(1,self.generator.num_ws,1), noise_mode=noise_mode)

            # classifier logit
            classifier_logit = self.classifier(normalize((synth_images-(synth_images.min()))/(synth_images.max()-synth_images.min()))) #self.classifier(synth_images) 

            # track images
            synth_images = (synth_images + 1) * (255/2)
            synth_images_np = synth_images.clone().detach().permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            all_images.append(synth_images_np)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # LPIPS loss
            synth_features = self.LM.vgg16(synth_images, resize_images=False, return_lpips=True)
            lpips_loss = (target_features - synth_features).square().sum()

            # MSE loss
            mse_loss = self.LM.l2_criterion(target_images, synth_images)

            # space regularizer
            num_of_sampled_latents = 1
            z_samples = np.random.randn(num_of_sampled_latents, self.generator.z_dim)
            z_samples = torch.from_numpy(z_samples).to(self.device)

            gen_c = self.LM.pti_loss.c_sampling_function(num_of_sampled_latents)
            reg_loss = self.LM.pti_loss.space_regularizer_loss(self.generator, G_pti, w_pivot, self.LM.vgg16, z_samples, gen_c)

            # Discriminator loss
            gen_logits = self.discriminator(synth_images, gen_c)
            loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])
            
            crossentropy_losses = self.LM.classifier_crossentropy_loss_on_generated_class_to_predict(classifier_logit, gen_c.argmax())

            ce_w = 0.01
            ds_w = 0.005
                
            # Step
            optimizer.zero_grad(set_to_none=True)
            loss = 0.1 * mse_loss + lpips_loss + reg_loss  + ce_w* crossentropy_losses + ds_w*loss_Dgen #+ crossentropy_losses
            loss.backward()
            optimizer.step()

            msg  = f'[ step {step+1:>4d}/{num_steps}] '
            msg += f'[ loss: {float(loss):<5.2f}] '
            msg += f'[ lpips: {float(lpips_loss):<5.2f}] '
            msg += f'[ mse: {float(mse_loss):<5.2f}]'
            msg += f'[ reg: {float(reg_loss):<5.2f}]'
            msg += f'[ disc: {float(loss_Dgen):<5.2f}]'
            msg += f'[ cross: {float(crossentropy_losses):<5.2f}]'
            if verbose: print(msg)

            
        return all_images, G_pti
