import shutil
import sys
import atexit
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, \
    QGridLayout, QScrollArea
from PyQt6.QtGui import QPixmap

latent_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def denorm(img_tensor):
    return img_tensor * stats[1][0] + stats[0][0]


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(

            nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AbsArt Generator")
        self.setFixedSize(128 * 4 + 60, 128 * 3 - 20)
        self.i = 0

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.image_widget = QWidget()
        self.image_layout = QGridLayout(self.image_widget)

        self.button = QPushButton("Generate an image")
        self.layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignBottom)

        self.images = []

        self.button.clicked.connect(self.generate)

    def generate(self):

        model = Generator()
        model.load_state_dict(torch.load('model/GEN00299.pt', map_location="cpu"))
        model.eval()

        sample_dir = "images/"
        fixed_latent = torch.randn(1, latent_size, 1, 1)
        fake_images = model(fixed_latent)

        path = os.path.join(sample_dir, f'fake_fname{self.i}.png')
        save_image(denorm(fake_images), path, nrow=1)
        self.i += 1

        self.images.append(path)

        self.display_images()

    def display_images(self):
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)

        for i, image_path in enumerate(self.images):
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(128, 128)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.image_layout.addWidget(label, i // 4, i % 4)  # Отображение в виде 4xN сетки

        self.scroll_area.setWidget(self.image_widget)


if __name__ == "__main__":
    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_list = os.listdir(directory)
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
