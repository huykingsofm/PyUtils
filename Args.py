import sys
class Args():
    def __init__(self, args : list):
        self.__args__ = args

    def GetFlag(self):
        Flags = []

        for arg in self.__args__: 
            if (arg[0] == "-" and arg[:2] != "--"):
                Flags.append(arg)

        return Flags

    def GetDefaultArgs(self):
        DefaultArgs = []
        for idx, arg in enumerate(self.__args__):
            if (arg[0] != "-"): # if arg is not flag or special parameters
                # if this arg is first argument or previous arg is not flag or special parmeters
                if (idx == 0 or self.__args__[idx - 1][:2] != "--"):
                    DefaultArgs.append(arg)
        return DefaultArgs

    def GetParameters(self):
        Parmeters = []

        for idx, arg in enumerate(self.__args__):
            if (arg[:2] == "--" and idx < len(self.__args__) - 1):
                Parmeters.append({"name" : arg[2:], "value" : self.__args__[idx + 1]})
        
        return Parmeters

if __name__ == "__main__":
    args = Args(sys.argv[1:])
    print(args.GetDefaultArgs())
    print(args.GetFlag())
    print(args.GetParameters())