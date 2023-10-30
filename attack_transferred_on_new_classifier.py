import json
from timm.models import load_checkpoint
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as md
from tqdm import tqdm

device = torch.device('cpu')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Resize((256,256)), transforms.CenterCrop((224,224)), 
    transforms.Normalize(mean, std)])

model_list = ["RegNetX","vit"] #["ResNet50_ImageNet_PRIME_noJSD.ckpt","fan_vit_base.pth.tar"]
model_type = ["vit","ResNet50"]
ckpt_type = ["ckpt", "tar"]
base_dir="/APT_DATA/"
dataset_list = ["prime_resnet50_val","prime_resnet50_val_fool_only"]


f= open(base_dir+"imagenet_class_index.json")
idxs = json.load(f)
inv_map = {v[0]: int(k) for k, v in idxs.items()}

def disable_gradient_flow_for_model(model:torch.nn.Module, device):
    model.eval().requires_grad_(False)
    model.to(device)

def get_base_model(name):
    if "vit" in name:
        return md.vit_l_16(pretrained=True)
    
    elif "ResNet50" in name:
        resnet = md.resnet50(pretrained=True)
        return resnet
    
    elif "RegNetX" in name:
        regnet = md.regnet_x_16gf(pretrained=True)
        return regnet


def get_model(name):
    #assert sum([tt in name for tt in model_type])==1

    return get_base_model(name)


def get_dataset(name):
    train_dataset = torchvision.datasets.ImageFolder(base_dir+name,transform=test_transform)

    return train_dataset


def eval_loop(model,train_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                            shuffle=False,
                                            num_workers=2,
                                            drop_last=True)
    model.eval()
    disable_gradient_flow_for_model(model,device)

    preds = []
    elements = 0
    failed = 0
    tricked = 0
    hacked = 0
    stayed_the_same = 0
    running_softmax_score = 0


    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
    #if should_be_the_same[batch_idx]:
        # classifier logit
        x, c = batch
        name= train_dataset.samples[batch_idx][0].split("/")[-2].split("_")[0]
        c = torch.tensor([inv_map[name]])
        x = x.to(device)
        c = c.to(device)

        classifier_logit = model(x)

        softmax_score_old = torch.nn.Softmax(dim=1)(classifier_logit)
        batch_idx
        for idx, elem in enumerate(c):
            
            score = (softmax_score_old)[idx,elem]
            elements +=1

            if (softmax_score_old[idx].argmax() != c[idx]).item():

                tricked += 1
                preds.append(1)

            if (softmax_score_old[idx].argmax() == c[idx]).item():

                stayed_the_same += 1
                preds.append(0)

            running_softmax_score += score
    running_softmax_score = running_softmax_score/elements
    acc= stayed_the_same/elements
    print(acc)
    print(running_softmax_score)
    data = {"running_score":running_softmax_score.item(), "acc":acc, "tricked":tricked, "stayed_the_same":stayed_the_same, "preds":preds}
    return data

for data in dataset_list:
    for mdl in model_list:
        model = get_model(mdl)
        dataset = get_dataset(data)
        result = eval_loop(model,dataset)

        savename = "results/"+data.split("/")[-1]+"_" + mdl.split("/")[-1].split(".")[0]+".json"
        with open(savename, 'w') as f:
            json.dump(result, f)
