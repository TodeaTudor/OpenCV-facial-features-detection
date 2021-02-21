from tkinter import *
from tkinter import filedialog
import os

root = Tk("Select mode")
root.geometry("200x50")
root.title("Select input mode")


def camera():
    os.system("python facial_detection.py 1")


def browse():
    filename = filedialog.askopenfilename()
    os.system("python facial_detection.py 0 " + filename)


browse_button = Button(root, text="Image", command=browse)
browse_button.pack()
camera_button = Button(root, text="Camera", command=camera)
camera_button.pack()
root.mainloop()
