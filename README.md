
# 🖐️ Hand Gesture Recognition System

A real-time **gesture control** project built using **OpenCV**, **MediaPipe**, and **NumPy** for intuitive interaction using hand gestures. Control your screen, draw, scroll, and more — with just your hands! 🖐️💻

---

## ✨ Features

- 🔽 **Scroll Down**  
  ➤ **Closed fist** (✊): All fingers down.

- 🔼 **Scroll Up**  
  ➤ **All fingers up except the thumb** (🖐️).

- 🔊 **Volume Control**  
  ➤ **Middle finger and thumb pinch** (🤏): Pinch to raise or lower the volume.

- 📸 **Take Screenshot**  
  ➤ **OK gesture** (👌): Index and thumb form a circle — **only in normal mode**.

- 🎨 **Toggle Drawing Mode**  
  ➤ **Index and pinky up** (🤘): Switch **drawing mode** on/off.

- 🌟 **Brightness Control**  
  ➤ **Thumb and pinky up** (🤙): Adjust brightness **only in normal mode**.

- 🖊️ **Pen Mode (Drawing)**  
  ➤ **OK gesture** (🖊️): Allows you to draw **only in drawing mode**.

---

## 🧰 Libraries Used

- 🔷 **[OpenCV](https://opencv.org/)** – Real-time video capture and image processing  
- ✋ **[MediaPipe](https://mediapipe.dev/)** – Hand tracking and landmark detection  
- 📐 **[NumPy](https://numpy.org/)** – Numerical calculations and array operations  
- 🪟 **[pywin32 / win32clipboard](https://pypi.org/project/pywin32/)** – Clipboard access and Windows screenshot support

---

## ✋ Gesture Cheat Sheet

| 🖼️ Gesture | 💥 Action |
|:----------:|:----------|
| ✊ **Closed fist** | 🔽 Scroll Down |
| 🖐️ **All fingers except thumb up** | 🔼 Scroll Up |
| 🤏 **Middle finger + Thumb Pinch** | 🔊 Volume Control |
| 👌 **OK Gesture** | 📸 Take Screenshot (Normal Mode) |
| 🤘 **Index + Pinky Up** | 🎨 Toggle Drawing Mode |
| 🤙 **Thumb + Pinky Up** | 🌟 Adjust Brightness / 🧹 Clear Canvas |

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

- ✂️ **Pinch-related functions** have been **removed** and **reserved for future use**.
- 🧠 Gestures are designed for **natural use**, **speed**, and **high reliability**.

---

## 💡 Built with ❤️ using OpenCV, MediaPipe & Python

---

Would you also like me to show how you can add the **gesture images** into the cheat sheet table too? 🖼️ It’ll look super polished and professional! 🎯  
(Example: adding your `.png` gesture icons next to each action!)  
Should I prepare that too? 🚀
