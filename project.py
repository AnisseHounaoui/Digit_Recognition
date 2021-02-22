import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
run = False



def draw(event, x, y, flag, param):
	global run

	if event == cv2.EVENT_LBUTTONDOWN:
		run = True
		cv2.circle(win, (x,y), 2 , (0,255,255), 2)

	if event == cv2.EVENT_LBUTTONUP:
		run = False

	if event == cv2.EVENT_MOUSEMOVE:
		if run == True:
			cv2.circle(win, (x,y), 2 , (0,255,255), 2)

	#if event == cv2.EVENT_RBUTTONDOWN:
		#action right button


#MODELE :

train_data = torchvision.datasets.MNIST(root='./MNIST', train=True,
										download=True, transform=transforms.ToTensor())

test_data = torchvision.datasets.MNIST(root='./MNIST', train=False,
									   download=True, transform=transforms.ToTensor())


def extraction(image):
	# split image to 16 cells
	for j in range(16):
		# feature 1
		# apply regression linear
		# feature 2
		# feature 3
		t=0
	return


BatchSize = 16
train, test = torch.utils.data.random_split(train_data, [50000, 10000])
train_loader = torch.utils.data.DataLoader(train, batch_size=BatchSize)  # Creating dataloader
test_loader = torch.utils.data.DataLoader(test, batch_size=BatchSize)

use_gpu = torch.cuda.is_available()
if use_gpu:
	#print('GPU is available!')
	device = "cuda"
else:
	#print('GPU is not available!')
	device = "cpu"


class MyNN(nn.Module):
	def __init__(self):
		super(MyNN, self).__init__()
		self.Layer1 = nn.Sequential(
			nn.Linear(28*28, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU())
		self.Layer2 = nn.Sequential(
			nn.Linear(32, 10))

	def forward(self, x):
		x = self.Layer1(x)
		x = self.Layer2(x)
		return x


network = MyNN()
network = network.to(device)

loss_fct1 = nn.CrossEntropyLoss()
optim_fct1 = optim.SGD(network.parameters(), lr=0.005)

def TrainModel(model, optim_fct, loss_fct, data_input, label):
	model.train()  # For training
	optim_fct.zero_grad()
	output = model(data_input)
	loss = loss_fct(output, label)
	loss.backward()
	optim_fct.step()
	return loss.item()


iterations = 5

for epoch in range(iterations):

	# training
	train_loss = 0
	# network.train()
	for data in train_loader:
		feature, label = data
		# output = network(torch.flatten(feature,start_dim=1))
		feature, label = feature.view(-1, 28 * 28).to(device), label.to(device)

		loss = TrainModel(network, optim_fct1, loss_fct1, feature, label)
		train_loss += loss
		train_loss /= len(train_loader)

	network.eval()
	print("epoch" + str(epoch+1))
	# testing
	with torch.no_grad():  # Gradient computation is not involved in inference
		test_loss = 0
		correct = 0
		total = 0
		for data in test_loader:
			feature, label = data
			feature, label = feature.view(-1, 28 * 28).to(device), label.to(device)
			total += label.size(0)
			output = network(feature)
			_, predicted = torch.max(output.data, 1)
			correct += (predicted == label).sum()
	print('At Epoch ' + str(epoch + 1))
	print('SGD: Loss = {:.6f} , Acc = {:.4f}'.format(train_loss, float(correct) * 100 / float(total)))


cv2.namedWindow('window')
cv2.setMouseCallback('window', draw)

win = np.zeros((250,250,3), dtype='float32')
while True:

	cv2.imshow('window', win)

	k = cv2.waitKey(1)

	if k == ord('c'):
		win = np.zeros((250,250,3), dtype='float32')

	if k == ord('q'):
		cv2.destroyAllWindows()
		break


	if k == ord("p"):
		image = win.copy()
		cv2.imwrite("./images/output.jpg",image)
		image = cv2.imread("./images/output.jpg")
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)

		# perform edge detection, find contours in the edge map, and sort the
		# resulting contours from left-to-right
		edged = cv2.Canny(blurred, 30, 150)
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sort_contours(cnts, method="left-to-right")[0]

		for c in cnts:
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# filter out bounding boxes, ensuring they are neither too small
			# nor too large
			if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
				# extract the character and threshold it to make the character
				# appear as *white* (foreground) on a *black* background, then
				# grab the width and height of the thresholded image
				roi = gray[y:y + h, x:x + w]
				thresh = cv2.threshold(roi, 0, 255,
									   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				(tH, tW) = thresh.shape
				# if the width is greater than the height, resize along the
				# width dimension
				if tW > tH:
					thresh = imutils.resize(thresh, width=32)
				# otherwise, resize along the height
				else:
					thresh = imutils.resize(thresh, height=32)
		out = cv2.resize(thresh,(28,28))
		cv2.imwrite("./images/feature.jpg", out)

		print(out.shape)# 28x28
		'''cv2.imshow("image", output_final)
		key = cv2.waitKey(0)
		cv2.imwrite("./images/cropped.jpg",roi)'''
		out = out.astype("float32")
		tens = torch.tensor(out)

		tens = tens.view(-1, 28*28) #.to(device)
		tens = tens / 255.0
		#print(tens)
		#print(tens.size())
		print(tens)
		output = network(tens)
		_,pred = torch.max(output.data, 1)
		print(output)
		print(pred)



#train_data[0][0].view(-1, 28 * 28)
#print(train_data[0][0].size())