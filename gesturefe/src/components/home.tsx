import { Link } from "react-router-dom";
export const Home = () => {
  return (
    <div className="px-80 pt-30 relative h-full w-full flex items-center justify-between">
      <div className="text-white pt-25 text-7xl/20 font-inter font-bold w-1/2">
        <div className="text-ascent">Control Everything with a Wave —</div>
        <div className="">Meet Gesture Flow.</div>
        <div className="text-sm pt-6 font-semibold text-white/80">
          Navigate your world with natural, intuitive gestures.
        </div>
        <div className="flex pt-10 gap-6">
          <Link to="/dashboard">
            <button className="px-6 text-2xl bg-white text-ascent p-4 rounded-full transition-transform active:scale-95 duration-200 ease-in-out hover:bg-white/90 hover:text-asdark">
              Try now!
            </button>
          </Link>
          <Link to="/about">
            <button className="text-2xl border-2 text-ascent p-4 px-6 rounded-full transition-transform active:scale-95 duration-100 ease-in-out hover:bg-aslight/10 hover:text-asdark">
              About
            </button>
          </Link>
        </div>
      </div>
      <div className="pt-35 pl-10">
        <img src="/landing.png" alt="landing" />
      </div>
    </div>
  );
};
