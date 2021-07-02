import torch, torchvision

# Load a pretrained resnet18 model
model = torchvision.models.resnet18(pretrained=True)

# Random tensor data to represent an image with three channels and height/width of 64
data = torch.rand(1,3,64,64)
# Image's label init'd to some random values
labels = torch.rand(1,1000)

# Run the input data through the model to make a prediction
prediction = model(data) # forward pass

# Calculate the error and backpropagate through the network
loss = (prediction - labels).sum()
loss.backward() # backward pass

# Load an optimizer with a learning rate of 0.01 and momentum of 0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Initiate gradient descent
optim.step() # gradient descent