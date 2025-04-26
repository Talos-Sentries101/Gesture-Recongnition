import { useLocation } from "react-router-dom";
import { Blobb } from "./blob.tsx";
export default function Background() {
  const { pathname: path } = useLocation();
  if (path === "/draw") {
    return null;
  }
  return (
    <div>
      <div className="flex justify-center items-center h-screen w-screen fixed -z-50 bg-background">
        <div className="absolute -left-30 -top-40 rotate-12">
          <Blobb />
        </div>
        <div className="absolute -right-70 bottom-30 rotate-180">
          <Blobb />
        </div>
        <div className="absolute left-64 -bottom-60 rotate-45">
          <Blobb />
        </div>
        <div className="absolute">
          <Blobb />
        </div>
      </div>
    </div>
  );
}
