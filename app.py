import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms

class Model(nn.Module):
  def __init__(self,num_classes=6):
    super(Model,self).__init__()

    #input shape = (256,3,150,150)
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
    #shape = (256,12,150,150)
    self.bn1 = nn.BatchNorm2d(num_features=12)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2)
    #reduce the image size by a factor of 2
    #shape = (256,12,75,75)

    self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
    #shape = (256,20,75,75)
    self.relu2 = nn.ReLU()
    #shape = (256,20,75,72)

    self.conv3 = nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
    #shape = (256,32,75,75)
    self.bn3 = nn.BatchNorm2d(num_features=32)
    #shape = (256,32,75,75)
    self.relu3 = nn.ReLU()
    #shape = (256,32,75,75)

    self.fc = nn.Linear(in_features=32*75*75,out_features=num_classes)

    #Feed Forward function
  def forward(self,input):
    output = self.conv1(input)
    output = self.bn1(output)
    output = self.relu(output)

    output = self.pool(output)

    output = self.conv2(output)
    output = self.relu2(output)

    output = self.conv3(output)
    output = self.bn3(output)
    output = self.relu3(output)

    #Output will be in matrix form of dimensions - (256,32,75,75)

    output = output.view(-1,32*75*75)
    output = self.fc(output)

    return output

# Load the trained PyTorch model
model = Model(num_classes=6)  # Adjust the number of classes based on your model
model.load_state_dict(torch.load('Intel image classifier.pt'))
model.eval()

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
    image = transform(image).unsqueeze(0)
    return image


st.title("Classify Images of Nature!")
st.header("Upload an image and classify it!")
st.subheader("This app is based on the CNN architecture and can be used to classify images of nature.")
st.subheader("The model can classify images of the following classes:")
st.subheader("Building, Forest, Glacier, Mountain, Sea, Street")

classes = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    processed_image = preprocess_image(image)
    with torch.no_grad():
        output = model(processed_image)
    

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_class = torch.max(output, 1)

    predicted_class_name = classes[predicted_class.item()]
    
    class_icons = {
       'building': '🏢',
       'forest': '🌳',
       'glacier': '❄️',
       'mountain': '⛰️',
       'sea': '🌊',
       'street': '🛣️',}
    
    if predicted_class_name in class_icons:
       class_icon = class_icons[predicted_class_name]
       st.write(f"The image is a {predicted_class_name} {class_icon}")
    else:
        st.write(predicted_class_name)

