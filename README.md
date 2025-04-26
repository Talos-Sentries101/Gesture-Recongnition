

# 🖐️ Hand Gesture Recognition System

A real-time **gesture control** project built using **OpenCV**, **MediaPipe**, and **NumPy** for intuitive interaction using hand gestures. Control your screen, draw, scroll, and more — with just your hands!

---

## ✨ Features

- 🔽 **Scroll Down**  
  ➤ **Closed fist** (✊): All fingers down.

- 🔼 **Scroll Up**  
  ➤ **All fingers up except the thumb** (🖐️).

- 🔊 **Volume Control**  
  ➤ **Middle finger and thumb pinch** (🤏): Pinch to raise/lower volume.

- 📸 **Take Screenshot**  
  ➤ **OK gesture** (👌): Index and thumb form a circle — **only in normal mode**.

- 🎨 **Toggle Drawing Mode**  
  ➤ **Index and pinky up** (🤘): Switch **drawing mode** on/off.

- 🧹 **Clear Canvas**  
  ➤ **Thumb and pinky up** (🤙): Clear the current drawing **only in drawing mode**.

---

## 🧰 Libraries Used

- 🔷 **[OpenCV](https://opencv.org/)** – Real-time video capture and image processing  
- ✋ **[MediaPipe](https://mediapipe.dev/)** – Hand tracking and landmark detection  
- 📐 **[NumPy](https://numpy.org/)** – Numerical calculations and array ops  
- 🪟 **[pywin32 / win32clipboard](https://pypi.org/project/pywin32/)** – Clipboard and Windows screenshot support



---

## ✋ Gesture Cheat Sheet

| 🖼️ Gesture | 💥 Action |
|:----------:|:----------|
| ✊ **Closed fist** | 🔽 Scroll down |
| 🖐️ **All except thumb up** | 🔼 Scroll up |
| 🤏 **Middle + thumb pinch** | 🔊 Volume control |
| 👌 **OK symbol** | 📸 Screenshot (normal mode only) |
| 🤘 **Index + pinky up** | 🎨 Toggle draw mode |
| 🤙 **Thumb + pinky up** | 🧹 Clear canvas |

---

## ⚙️ Setup Instructions

```bash
pip install opencv-python mediapipe numpy pywin32
```

▶️ Then run your main script:

```bash
python your_main_script.py
```

📷 **Make sure your webcam is connected and accessible!**

---

## 🗒️ Notes

- ✂️ **Pinch-related functions** are **removed** and **reserved for future gestures**.
- 🧠 Gestures are designed for **natural use**, **speed**, and **reliability**.

---

## 💡 Built with ❤️ using OpenCV, MediaPipe & Python

