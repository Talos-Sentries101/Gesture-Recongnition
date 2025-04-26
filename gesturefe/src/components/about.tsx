export const About = () => {
  return (
    <div className="text-white px-80 pt-30 flex flex-col relative h-full w-full gap-6">
      <Heading text="Made by The Byte Force" />
      <SubHeading text="Members" space={0} />
      <div className="cursor-pointer pl-10 text-xl text-white/80 text-bold font-inter flex flex-col ga-2">
        <a
          href="https://github.com/rakshith-reddy-23"
          target="_blank"
          className="underline"
        >
          T. Rakshith Reddy
        </a>
        <a
          href="https://github.com/mekala-2405"
          target="_blank"
          className="underline"
        >
          M. Harshvardhan Reddy
        </a>
        <a
          href="https://github.com/Talos-Sentries101"
          target="_blank"
          className="underline"
        >
          G. Srinikesh Reddy
        </a>
        <a
          href="https://github.com/ayushwasnothere"
          target="_blank"
          className="underline"
        >
          Ayush Kumar Shah
        </a>
      </div>
      <Heading text="About Project" />
      <Text
        text="A real-time gesture control project built using OpenCV, MediaPipe, and NumPy for intuitive interaction using hand gestures. Control your screen, draw, scroll, and more — with just your hands! 🖐️💻"
        space={0}
      />
      <Heading text="Features" />
      <Text
        text={`
🔽 Scroll Down
\n
➤ Closed fist (✊): All fingers down.`}
        space={0}
      />
      <Text
        text={`
🔼 Scroll Up
➤ All fingers up except the thumb (🖐️).
`}
        space={0}
      />
      <Text
        text={`
🔊 Volume Control
➤ Middle finger and thumb pinch (🤏): Pinch to raise or lower the volume.
`}
        space={0}
      />
      <Text
        text={`
📸 Take Screenshot
➤ OK gesture (👌): Index and thumb form a circle — only in normal mode.
`}
        space={0}
      />
    </div>
  );
};

const Heading = ({ text }: { text: string }) => {
  return (
    <div className="text-4xl inline font-inter font-bold text-white/80 border-b-4 py-1 border-ascent w-fit">
      {text} :
    </div>
  );
};

const SubHeading = ({ text, space }: { text: string; space: number }) => {
  return (
    <div
      className={`ml-${String(10 + (space || 0) * 5)} text-2xl inline font-inter font-bold text-white/80 border-b-4 py-1 border-ascent w-fit`}
    >
      {text} :
    </div>
  );
};

const Text = ({ text, space }: { text: string; space: number }) => {
  return (
    <div
      className={`pl-${String(10 + (space || 0) * 5)} text-xl text-white/80`}
    >
      {text}
    </div>
  );
};
