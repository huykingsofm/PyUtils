import sys

class Args():
    ALL = "all"
    ONLY = "only"
    NO_APPEAR = "no appear"
    SOME = "some but all"
    def __init__(self, args : list, flagnames : list, parameternames : list):
        self.__args__ = args
        self.__flagnames__ = flagnames
        self.__parameternames__ = parameternames

        names = []
        names.extend(flagnames)
        names.extend(parameternames)
        
        self.__splitposition__ = len(flagnames)

        self.__uniquenames__ = self.__get_unique_names__(names)

        self.__rename__(names, self.__uniquenames__)        
        self.parameters = self.__get_parameters__(self.__uniquenames__[self.__splitposition__ : ]) 
        self.flags = self.__get_flags__(self.__uniquenames__[ : self.__splitposition__])   
        self.default_args = self.__get_default_args__(self.__uniquenames__[self.__splitposition__ : ])    

    def __get_parameters__(self, parameternames : list):
        parameters = {}
        for idx, arg in enumerate( self.__args__ ):
            if (arg[:2] == "--" and arg[2:] in parameternames):
                if (idx + 1 >= len(self.__args__) or self.__args__[idx + 1][:2] == "--"):
                    raise Exception("{} must be have a value".format(arg[2:]))
                else:
                    parameters.update({arg[2:] : self.__args__[idx + 1]})

        return parameters
    
    def __get_unique_names__(self, names):
        uniquenames = []
        for name in names:
            newname = name[0] if name[0] != None else name[1]
            if (newname == None):
                raise Exception("Error in input names")
            uniquenames.append(newname)
        return uniquenames

    def __get_default_args__(self, parameternames : list):
        default_args = []
        for idx, arg in enumerate(self.__args__):
            if (arg[:2] != "--"):
                # if this arg is first argument or previous arg is not flag or special parmeters
                if (idx == 0 or self.__args__[idx - 1][2:] not in parameternames):
                    default_args.append(arg)
        return default_args
    
    def __get_flags__(self, flagnames : list):
        flags = []
        for arg in self.__args__:
            if (arg[:2] == "--" and arg[2:] in flagnames):
                flags.append(arg[2:])

        return flags

    def __rename__(self, argnames, newnames):
        if len(argnames) != len(newnames):
            raise Exception("Length of old names must be same as length of new names")

        existed = [False] * len(newnames)
        for iarg, arg in enumerate(self.__args__):
            if arg[:1] != "-":
                continue
            flag = False
            for i, name in enumerate( argnames ):
                if (arg[:2] == "--" and arg[2:] == name[0]):
                    if existed[i] == True:
                        raise Exception("Error in arguments, {} is defined before".format(arg))
                    existed[i] = True
                    flag = True
                    self.__args__[iarg] = "--" + newnames[i]

                elif (arg[:1] == "-" and arg[1:] == name[1]):
                    if existed[i] == True:
                        raise Exception("Error in arguments, {} is defined before".format(arg))
                    existed[i] = True
                    flag = True
                    self.__args__[iarg] = "--" + newnames[i]
            if not flag:
                raise Exception("Argument {} is not defined".format(arg))

    def CheckFlagSet(self, flagnames : list):
        for flagname in flagnames:
            if flagname not in self.__flagnames__:
                raise Exception("Flag {} is not defined".format(flagname))

        flagnames = self.__get_unique_names__(flagnames)
        count = 0
        for flagname in flagnames:
            count += 1 if flagname in self.flags else 0

        if count == len(flagnames):
            return Args.ALL
        
        if (count == 1):
            return Args.ONLY
        
        if count == 0:
            return Args.NO_APPEAR
        
        # some args appear
        return Args.SOME

    def CheckParameterSet(self, parameternames : list):
        for parametername in parameternames:
            if parametername not in self.__parameternames__:
                raise Exception("Parameter {} is not defined".format(parametername))

        parameternames = self.__get_unique_names__(parameternames)
        count = 0
        for parametername in parameternames:
            count += 1 if parametername in self.parameters.keys() else 0

        if count == len(parameternames):
            return Args.ALL
        
        if (count == 1):
            return Args.ONLY
        
        if count == 0:
            return Args.NO_APPEAR
        
        # some args appear
        return Args.SOME
                

if __name__ == "__main__":
    args = Args(sys.argv[1:], [["help", "h"], ["pthread", None]], [[None, "k"], ["iv", "i"]])
    print(args.__args__)
    print(args.parameters)
    print(args.flags)
    print(args.default_args)
    print(args.CheckFlagSet([["help", "h"], ["pthread", None]]))
    print(args.CheckParameterSet([[None, "k"], ["iv", "i"]]))