import { Camera, CircleSlash, LayoutDashboard, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export const Dashboard = () => {
  const videoRef = useRef<HTMLImageElement | null>(null);
  const [status, setStatus] = useState<
    "idle" | "connecting" | "connected" | "error"
  >(videoRef.current ? "connected" : "idle");
  const [gesture, setGesture] = useState("brightness_eraser");
  const [gestureName, setGestureName] = useState("None");

  useEffect(() => {
    switch (gesture) {
      case "brightness_eraser":
        setGestureName("Brightness Controller");
        break;
      case "double_tap":
        setGestureName("Left Click");
        break;
      case "drawing_mode_on_off":
        setGestureName("Toggle Draw");
        break;
      case "mouse_tracking":
        setGestureName("Mouse Pointer");
        break;
      case "screenshot_pencil":
        setGestureName("Take Screenshot");
        break;
      case "scroll_down":
        setGestureName("Scroll Down");
        break;
      case "scroll_up":
        setGestureName("Scroll Up");
        break;
      default:
        setGestureName("none");
    }
  }, [gesture]);
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:5000/camera_status")
        .then((res) => res.json())
        .then((data) => {
          setGesture(data.gesture);
          setStatus(data.status);
        })
        .catch((err) => {
          console.error("Failed to fetch camera status", err);
          setStatus("error");
        });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.src = "http://localhost:5000/video_feed";
    }
  }, []);

  return (
    <div className="w-screen h-screen flex flex-col 2xl:px-80 xl:px-60 px-10 relative pt-30">
      <div className="h-full relative flex flex-col">
        <div className="py-5 flex items-center gap-2 text-white font-bold font-inter text-3xl">
          <LayoutDashboard className="h-8 w-8 text-aslight" />
          Live Preview
        </div>
        <div className="flex bg-black/30 rounded-t-xl backdrop-blur-[10px] shadow-4xl overflow-hidden">
          <div className="h-[480px] w-[720px] bg-black flex justify-center items-center relative flex-col rounded-t-xl">
            {status === "idle" && (
              <div className="text-center">
                <Camera className="h-12 w-12 text-aslight mx-auto mb-2" />
                <p className="text-gray-400">Camera disabled</p>
              </div>
            )}

            {status === "connecting" && (
              <div className="text-center">
                <RefreshCw className="h-10 w-10 text-aslight mx-auto mb-2 animate-spin" />
                <p className="text-aslight">Connecting to camera...</p>
              </div>
            )}

            {status === "connected" && (
              <div>
                <img ref={videoRef} alt="Video feed" />
              </div>
            )}
          </div>
          <div className="flex flex-col justify-center items-center mx-auto">
            {gesture == "none" ? (
              <div className="flex flex-col justify-center items-center gap-4 text-white font-bold font-inetr text-xl mx-auto">
                <CircleSlash className="text-ascent h-20 w-20" />
                <div>No gesture detected</div>
              </div>
            ) : (
              <div className="flex flex-col justify-center items-center gap-4 text-white font-bold font-inetr text-xl mx-auto">
                <img
                  src={`/public/gestures/${gesture}.png`}
                  className="text-ascent h-40 w-40"
                />
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col py-6 px-6 bg-black/30 gap-4 rounded-b-xl backdrop-blur-[10px] shadow-4xl overflow-hidden">
          <div className="flex justify-between items-center w-full px-4">
            <div className="text-white font-bold text-2xl">Gesture Preview</div>
            <div className="w-20 flex justify-center items-center"></div>
          </div>
          <div className="flex gap-5 w-full">
            <div className="p-4 bg-background/80 rounded-xl w-full">
              <div className="text-white/60 text-sm">Status</div>
              <div className="text-white text-lg font-semibold">Camers on</div>
            </div>
            <div className="p-4 bg-background rounded-xl w-full">
              <div className="text-white/60 text-sm">Current gesture</div>
              <div className="text-white text-lg font-semibold">
                {gestureName}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
