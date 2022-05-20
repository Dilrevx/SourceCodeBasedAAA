import os
import re
import sys
import javalang
from javalang.tokenizer import tokenize
from javalang.parser import Parser
import javalang.tree
from lxml import etree


class JavaFile(object):
    def __init__(self, dec_path, typ):
        super(JavaFile, self).__init__()
        self.java_folder_path = dec_path + "/sources"
        # self.java_files, self.library_feature = self.getJavaFiles(typ)
        self.java_files = self.getCustomJavaFiles(typ)

    def getPackageName(self, dec_path):
        tree = etree.ElementTree(file="%s/AndroidManifest.xml" % dec_path)
        root = tree.getroot()
        packageName = root.attrib['package'].replace('.', '/')
        return packageName

    def getCustomJavaFiles(self, typ):
        java_files = []
        library_feature = {}
        packageName = self.getPackageName(os.path.dirname(self.java_folder_path.replace("Java", "Smali")))

        for root, subs, files in os.walk("%s" % self.java_folder_path):
            for filename in files:
                if int(typ) == 0:
                    if packageName in root:
                        java_files.append(os.path.join(root, filename))
                    # else:
                    #    library_feature[root.split('/smali/')[-1]] = 1
                    # print(root)
                else:
                    java_files.append(os.path.join(root, filename))
        return java_files

    # type 0 -> custom codes
    # type 1 -> all codes including library codes
    def getJavaFiles(self, typ):
        java_files = []
        libraries = []
        library_feature = {}

        with open(os.getcwd() + "/Data/library_list", 'r+') as f:
            lineno = 0
            for line in f:
                lineno += 1
                libraries.append(line[:-1])

        for lib in libraries:
            library_feature[lib] = 0

        if int(typ) == 0:
            # libs_name = lrd.tree.getLibs()
            for root, subs, files in os.walk(self.java_folder_path):
                isLib = False
                for lib in libraries:
                    # if lib in root.split('smali/')[-1]:
                    if root.split('sources/')[-1] == "com/android":
                        aaaa = 5
                    if root.split('sources/')[-1].__eq__(lib):
                        isLib = True
                        library_feature[lib] = 1  
                        break
                if isLib == False:
                    for filename in files:
                        java_files.append(os.path.join(root, filename))
            return java_files, library_feature
        else:
            for root, subs, files in os.walk(self.java_folder_path):
                for lib in libraries:
                    if lib in root:
                        library_feature[lib] = 1
                        break
                for filename in files:
                    java_files.append(os.path.join(root, filename))
            return java_files, library_feature

    @property
    def parse(self):
        java_feature = {'avgCharPerLine': 0,
                        'avgCharPerLocalVar': 0,
                        'avgCharPerGlobalVar': 0,
                        'avgCharPerFuncName': 0,
                        'avgCharPerFuncParamName': 0,
                        'avgLinePerClass': 0,
                        'avgLinePerFunc': 0,
                        'avgFuncPerClass': 0,
                        'ratioGlobalVarToAllCodes': 0,
                        'ratioLocalVarToAllCodes': 0,
                        'ratioVarToAllCodes': 0,
                        'ratioIfToAllCodes': 0,
                        'ratioInvokeToAllCodes': 0,
                        'ratioIntToAllFunc': 0,
                        'ratioVoidToAllFunc': 0,
                        'ratioUpperToAllFunc': 0,
                        'ratioUpperToAllVar': 0,
                        'ratioMoveToAllCodes': 0,
                        }

        number_of_line = 0
        number_of_func_line = 0
        number_of_local_var = 0
        number_of_global_var = 0
        number_of_upper_var = 0
        number_of_func = 0
        number_of_upper_func = 0
        number_of_param = 0
        number_of_if = 0
        number_of_invoke = 0
        number_of_move = 0
        number_of_I_return_type = 0
        number_of_V_return_type = 0
        number_of_total_function = 0
        number_of_total_function_param = 0
        codes = []

        number_of_total_character = 0
        number_of_total_local_var_character = 0
        number_of_total_global_var_character = 0
        number_of_total_function_character = 0
        number_of_total_function_param_character = 0
        number_of_total_line_all_function = 0
        try:
            for java_file in self.java_files:
                with open(java_file, "r") as file:
                    source = file.read()
                    number_of_total_character += source.replace(" ", "").replace('\n', '').__len__()
                    number_of_line += list(filter(None, source.split('\n'))).__len__()

                    for line in source.split('\n'):
                        if re.search('if (.*)', line):
                            number_of_if += 1

                    try:
                        tokens = tokenize(source)
                        parser = Parser(tokens)
                        ast = parser.parse()

                        for method in ast.types[0].methods:
                            if method.return_type:
                                if method.return_type.name == 'int':
                                    number_of_I_return_type += 1
                            else:
                                # if method.return_type.name == 'void':
                                number_of_V_return_type += 1
                            if method.name[0].isupper():
                                number_of_upper_func += 1
                            number_of_total_function += 1
                            number_of_total_function_character += len(method.name)
                            for p in method.parameters:
                                if isinstance(p, javalang.tree.FormalParameter):
                                    number_of_total_function_param += 1
                                    number_of_total_function_param_character += len(p.name)
                            if method.body:
                                number_of_total_line_all_function += method.body[-1].position.line - method.position.line + 2
                                for f in method.body:
                                    if isinstance(f, javalang.tree.LocalVariableDeclaration):  # localvariables
                                        for lv in f.declarators:
                                            if lv.name[0].isupper():
                                                number_of_upper_var += 1
                                            number_of_local_var += 1
                                            number_of_total_local_var_character += len(lv.name)

                        for field in ast.types[0].fields:  # globalvariable
                            if field.declarators[0].name[0].isupper():
                                number_of_upper_var += 1
                            number_of_global_var += 1
                            number_of_total_global_var_character += len(field.declarators[0].name)
                    except:
                        err = str(sys.exc_info()[0]) + str(sys.exc_info()[1])
                        # print(err)
                        print(err, java_file)

        except:
            err = str(sys.exc_info()[0]) + str(sys.exc_info()[1])
            print(err)
            print(java_file)

        if number_of_line != 0:
            java_feature["avgCharPerLine"] = number_of_total_character / float(number_of_line)
            java_feature["ratioGlobalVarToAllCodes"] = number_of_global_var / float(number_of_line)
            java_feature["ratioLocalVarToAllCodes"] = number_of_local_var / float(number_of_line)
            java_feature["ratioVarToAllCodes"] = (number_of_global_var + number_of_local_var) / float(number_of_line)
            java_feature["ratioIfToAllCodes"] = number_of_if / float(number_of_line)

        if number_of_total_function != 0:
            java_feature["avgCharPerFuncName"] = number_of_total_function_character / float(number_of_total_function)
            java_feature["avgLinePerFunc"] = number_of_total_line_all_function / float(number_of_total_function)
            java_feature["ratioUpperToAllFunc"] = number_of_upper_func / float(number_of_total_function)
            java_feature["ratioIntToAllFunc"] = float(number_of_I_return_type) / number_of_total_function
            java_feature["ratioVoidToAllFunc"] = float(number_of_V_return_type) / number_of_total_function

        if number_of_local_var != 0:
            java_feature["avgCharPerLocalVar"] = number_of_total_local_var_character / float(number_of_local_var)
        if number_of_global_var != 0:
            java_feature["avgCharPerGlobalVar"] = number_of_total_global_var_character / float(number_of_global_var)

        if (number_of_local_var + number_of_global_var) != 0:
            java_feature["ratioUpperToAllVar"] = number_of_upper_var / float((number_of_local_var + number_of_global_var))

        if number_of_total_function_param != 0:
            java_feature["avgCharPerFuncParamName"] = number_of_total_function_param_character / float(number_of_total_function_param)

        if len(self.java_files) != 0:
            java_feature["avgLinePerClass"] = float(number_of_line) / len(self.java_files)
            java_feature["avgFuncPerClass"] = float(number_of_total_function) / len(self.java_files)

        return java_feature

    @property
    def parseJava(self):
        java_feature = {'avgCharPerLine': 0,
                        'avgCharPerLocalVar': 0,
                        'avgCharPerGlobalVar': 0,
                        'avgCharPerFuncName': 0,
                        'avgCharPerFuncParamName': 0,
                        'avgLinePerClass': 0,
                        'avgLinePerFunc': 0,
                        'avgFuncPerClass': 0,
                        'ratioGlobalVarToAllCodes': 0,
                        'ratioLocalVarToAllCodes': 0,
                        'ratioVarToAllCodes': 0,
                        'ratioIfToAllCodes': 0,
                        'ratioInvokeToAllCodes': 0,
                        'ratioIntToAllFunc': 0,
                        'ratioVoidToAllFunc': 0,
                        'ratioUpperToAllFunc': 0,
                        'ratioUpperToAllVar': 0,
                        'ratioMoveToAllCodes': 0,
                        }

        number_of_line = 0
        number_of_func_line = 0
        number_of_local_var = 0
        number_of_global_var = 0
        number_of_upper_var = 0
        number_of_func = 0
        number_of_upper_func = 0
        number_of_param = 0
        number_of_if = 0
        number_of_invoke = 0
        number_of_move = 0
        number_of_I_return_type = 0
        number_of_V_return_type = 0
        codes = []
        try:
            for java_file in self.java_files:
                with open(java_file, "r") as file:
                    copy = False
                    for line in file:
                        code = line.strip()
                        if code != "" and not code.startswith("#"):
                            codes.append(code)
                            number_of_line += 1
                            if code.startswith(".method"):  # average Char per function name
                                copy = True
                                return_type = code.split(" ")[-1].split(")")[1]
                                if return_type.__eq__("I"):
                                    number_of_I_return_type += 1
                                elif return_type.__eq__("V"):
                                    number_of_V_return_type += 1
                                function_name = code.split(" ")[-1].split("(")[0]
                                match = re.search("\<\\w*init\>", function_name)
                                if not match:
                                    java_feature["avgCharPerFuncName"] = float(
                                        java_feature["avgCharPerFuncName"] * number_of_func + len(function_name)) / (number_of_func + 1)
                                    if function_name.isupper():
                                        number_of_upper_func += 1
                                    number_of_func += 1
                                continue
                            elif ".end method" in code:
                                copy = False
                                continue
                            elif code.startswith(".param"):  # average number of Function Parameter Name
                                param_name = code.split(",")[-1].split("#")[0].strip()[1:-1]
                                match = re.search("this\$", param_name)
                                if not match:
                                    java_feature["avgCharPerFuncParamName"] = float(java_feature["avgCharPerFuncParamName"] * number_of_param + len(param_name)) / (number_of_param + 1)
                                    number_of_param += 1
                                continue
                            elif code.startswith(".field"):  # avg char per global var
                                glob = code.split(":")[0].split(" ")[-1]
                                java_feature["avgCharPerGlobalVar"] = float(
                                    java_feature["avgCharPerGlobalVar"] * number_of_global_var + len(glob)) / (number_of_global_var + 1)
                                if glob.isupper():
                                    number_of_upper_var += 1
                                number_of_global_var += 1
                                continue
                            elif code.startswith(".local"):  # avg char per local var
                                spl = code.split(",")
                                if len(spl) > 1:
                                    local = spl[1].split(":")[0][2:-1]
                                    java_feature["avgCharPerLocalVar"] = float(
                                        java_feature["avgCharPerLocalVar"] * number_of_local_var + len(local)) / (number_of_local_var + 1)
                                    if local.isupper():
                                        number_of_upper_var += 1
                                    number_of_local_var += 1
                                continue
                            elif code.startswith("if-"):
                                number_of_if += 1
                                continue
                            elif code.startswith("invoke-"):
                                number_of_invoke += 1
                                continue
                            elif code.startswith("move-"):
                                number_of_move += 1
                                continue
                            if copy:
                                number_of_func_line += 1
                                continue

            if number_of_line != 0:
                java_feature["avgCharPerLine"] = float(codes.__len__()) / number_of_line
                java_feature["ratioGlobalVarToAllCodes"] = float(number_of_global_var) / number_of_line
                java_feature["ratioLocalVarToAllCodes"] = float(number_of_local_var) / number_of_line
                java_feature["ratioVarToAllCodes"] = float(number_of_local_var + number_of_global_var) / number_of_line
                java_feature["ratioIfToAllCodes"] = float(number_of_if) / number_of_line
                java_feature["ratioInvokeToAllCodes"] = float(number_of_invoke) / number_of_line
                java_feature["ratioMoveToAllCodes"] = float(number_of_move) / number_of_line

            if number_of_func != 0:
                java_feature["avgLinePerFunc"] = float(number_of_func_line) / number_of_func
                java_feature["ratioIntToAllFunc"] = float(number_of_I_return_type) / number_of_func
                java_feature["ratioVoidToAllFunc"] = float(number_of_V_return_type) / number_of_func
                java_feature["ratioUpperToAllFunc"] = float(number_of_upper_func) / number_of_func

            if len(self.java_files) != 0:
                java_feature["avgLinePerClass"] = float(number_of_line) / len(self.java_files)
                java_feature["avgFuncPerClass"] = float(number_of_func) / len(self.java_files)

            if (number_of_local_var + number_of_global_var) != 0:
                java_feature["ratioUpperToAllVar"] = float(number_of_upper_var) / (number_of_local_var + number_of_global_var)


        except:
            err = str(sys.exc_info()[0]) + str(sys.exc_info()[1])
            print(err)

        return java_feature
