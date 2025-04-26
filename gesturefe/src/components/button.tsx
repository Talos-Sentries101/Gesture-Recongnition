interface ButtonProps {
  name: string;
  onClick: () => void;
}
export const CustomButton = ({ name, onClick }: ButtonProps) => {
  return (
    <button
      onClick={onClick}
      className="hover:bg-ascent/50 w-full py-2 text-base rounded-full font-bold text-white bg-ascent active:scale-95 transition-transform duration-200 ease-in-out flex items-center justify-center gap-2"
    >
      {name}
    </button>
  );
};
