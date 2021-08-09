##################################################################################
class CNN714(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 42 * 42, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)  # 712, 712 -> 356, 356
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)  # 354, 354 -> 177, 177
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)  # 175, 175 -> 87, 87
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, 2)  # 85, 85 -> 42, 42
        # print(x.size())
        x = x.view(-1, 64 * 42 * 42)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
