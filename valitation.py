import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import HandWashModel





def valitation(model_name, input, use_gpu):

    # model
    model = HandWashModel()
    model.load_state_dict(torch.load(model_name))

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    if use_gpu:
        model = model.to(device)
        input = input.to(device)

    model.eval()
    output = model.forward(input)
    output = output.max(1)
    return output


def get_tensor_from_video(video_path):
    if not os.access(video_path, os.F_OK):
        print(video_path, 'file does not exist')
        return

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(video_path)
    frame_rate = 25
    c = 0

    image_list = []
    while(cap.isOpened()):
        ret,frame = cap.read()

        if not ret:
            break
        else:
            if(c % frame_rate == 0):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image.save("./tmp/test_"+str(c)+".ppm")
                image = val_transforms(image)
                image_list.append(image)
            c += 1
    cap.release()
    # image_list[0].save("test.ppm")
    # result_frames = torch.as_tensor(np.stack(image_list))
    # print(image_list[0].size())
    result_frames = torch.stack(image_list)
    print(result_frames.size())
    return result_frames


def video_valitation(video_name):
    frames = get_tensor_from_video(video_name)

    output = valitation("./handwash_41.pth", frames, True)
    print(output)

def main():
    video_valitation("/mnt/nvme2n1/handwash_dataset/f11_18_lyz.mp4")

if __name__ == "__main__":
    main()
