import os
import re
import sys

from lxml import etree


# ALL_PERMISSIONS = [
#     "android.permission.ACCESS_CHECKIN_PROPERTIES",
#     "android.permission.ACCESS_COARSE_LOCATION",
#     "android.permission.ACCESS_FINE_LOCATION",
#     "android.permission.ACCESS_LOCATION_EXTRA_COMMANDS",
#     "android.permission.ACCESS_NETWORK_STATE",
#     "android.permission.ACCESS_NOTIFICATION_POLICY",
#     "android.permission.ACCESS_WIFI_STATE",
#     "android.permission.ACCOUNT_MANAGER",
#     "android.permission.ADD_VOICEMAIL",
#     "android.permission.ANSWER_PHONE_CALLS",
#     "android.permission.BATTERY_STATS",
#     "android.permission.BIND_ACCESSIBILITY_SERVICE",
#     "android.permission.BIND_APPWIDGET",
#     "android.permission.BIND_AUTOFILL_SERVICE",
#     "android.permission.BIND_CARRIER_MESSAGING_SERVICE",
#     "android.permission.BIND_CARRIER_SERVICES",
#     "android.permission.BIND_CHOOSER_TARGET_SERVICE",
#     "android.permission.BIND_CONDITION_PROVIDER_SERVICE",
#     "android.permission.BIND_DEVICE_ADMIN",
#     "android.permission.BIND_DREAM_SERVICE",
#     "android.permission.BIND_INCALL_SERVICE",
#     "android.permission.BIND_INPUT_METHOD",
#     "android.permission.BIND_MIDI_DEVICE_SERVICE",
#     "android.permission.BIND_NFC_SERVICE",
#     "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE",
#     "android.permission.BIND_PRINT_SERVICE",
#     "android.permission.BIND_QUICK_SETTINGS_TILE",
#     "android.permission.BIND_REMOTEVIEWS",
#     "android.permission.BIND_SCREENING_SERVICE",
#     "android.permission.BIND_TELECOM_CONNECTION_SERVICE",
#     "android.permission.BIND_TEXT_SERVICE",
#     "android.permission.BIND_TV_INPUT",
#     "android.permission.BIND_VISUAL_VOICEMAIL_SERVICE",
#     "android.permission.BIND_VOICE_INTERACTION",
#     "android.permission.BIND_VPN_SERVICE",
#     "android.permission.BIND_VR_LISTENER_SERVICE",
#     "android.permission.BIND_WALLPAPER",
#     "android.permission.BLUETOOTH",
#     "android.permission.BLUETOOTH_ADMIN",
#     "android.permission.BLUETOOTH_PRIVILEGED",
#     "android.permission.BODY_SENSORS",
#     "android.permission.BROADCAST_PACKAGE_REMOVED",
#     "android.permission.BROADCAST_SMS",
#     "android.permission.BROADCAST_STICKY",
#     "android.permission.BROADCAST_WAP_PUSH",
#     "android.permission.CALL_PHONE",
#     "android.permission.CALL_PRIVILEGED",
#     "android.permission.CAMERA",
#     "android.permission.CAPTURE_AUDIO_OUTPUT",
#     "android.permission.CAPTURE_SECURE_VIDEO_OUTPUT",
#     "android.permission.CAPTURE_VIDEO_OUTPUT",
#     "android.permission.CHANGE_COMPONENT_ENABLED_STATE",
#     "android.permission.CHANGE_CONFIGURATION",
#     "android.permission.CHANGE_NETWORK_STATE",
#     "android.permission.CHANGE_WIFI_MULTICAST_STATE",
#     "android.permission.CHANGE_WIFI_STATE",
#     "android.permission.CLEAR_APP_CACHE",
#     "android.permission.CONTROL_LOCATION_UPDATES",
#     "android.permission.DELETE_CACHE_FILES",
#     "android.permission.DELETE_PACKAGES",
#     "android.permission.DIAGNOSTIC",
#     "android.permission.DISABLE_KEYGUARD",
#     "android.permission.DUMP",
#     "android.permission.EXPAND_STATUS_BAR",
#     "android.permission.FACTORY_TEST",
#     "android.permission.FOREGROUND_SERVICE",
#     "android.permission.GET_ACCOUNTS",
#     "android.permission.GET_ACCOUNTS_PRIVILEGED",
#     "android.permission.GET_PACKAGE_SIZE",
#     "android.permission.GET_TASKS",
#     "android.permission.GLOBAL_SEARCH",
#     "android.permission.INSTALL_LOCATION_PROVIDER",
#     "android.permission.INSTALL_PACKAGES",
#     "android.permission.INSTALL_SHORTCUT",
#     "android.permission.INSTANT_APP_FOREGROUND_SERVICE",
#     "android.permission.INTERNET",
#     "android.permission.KILL_BACKGROUND_PROCESSES",
#     "android.permission.LOCATION_HARDWARE",
#     "android.permission.MANAGE_DOCUMENTS",
#     "android.permission.MANAGE_OWN_CALLS",
#     "android.permission.MASTER_CLEAR",
#     "android.permission.MEDIA_CONTENT_CONTROL",
#     "android.permission.MODIFY_AUDIO_SETTINGS",
#     "android.permission.MODIFY_PHONE_STATE",
#     "android.permission.MOUNT_FORMAT_FILESYSTEMS",
#     "android.permission.MOUNT_UNMOUNT_FILESYSTEMS",
#     "android.permission.NFC",
#     "android.permission.NFC_TRANSACTION_EVENT",
#     "android.permission.PACKAGE_USAGE_STATS",
#     "android.permission.PERSISTENT_ACTIVITY",
#     "android.permission.PROCESS_OUTGOING_CALLS",
#     "android.permission.READ_CALENDAR",
#     "android.permission.READ_CALL_LOG",
#     "android.permission.READ_CONTACTS",
#     "android.permission.READ_EXTERNAL_STORAGE",
#     "android.permission.READ_FRAME_BUFFER",
#     "android.permission.READ_INPUT_STATE",
#     "android.permission.READ_LOGS",
#     "android.permission.READ_PHONE_NUMBERS",
#     "android.permission.READ_PHONE_STATE",
#     "android.permission.READ_SMS",
#     "android.permission.READ_SYNC_SETTINGS",
#     "android.permission.READ_SYNC_STATS",
#     "android.permission.READ_VOICEMAIL",
#     "android.permission.REBOOT",
#     "android.permission.RECEIVE_BOOT_COMPLETED",
#     "android.permission.RECEIVE_MMS",
#     "android.permission.RECEIVE_SMS",
#     "android.permission.RECEIVE_WAP_PUSH",
#     "android.permission.RECORD_AUDIO",
#     "android.permission.REORDER_TASKS",
#     "android.permission.REQUEST_COMPANION_RUN_IN_BACKGROUND",
#     "android.permission.REQUEST_COMPANION_USE_DATA_IN_BACKGROUND",
#     "android.permission.REQUEST_DELETE_PACKAGES",
#     "android.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
#     "android.permission.REQUEST_INSTALL_PACKAGES",
#     "android.permission.RESTART_PACKAGES",
#     "android.permission.SEND_RESPOND_VIA_MESSAGE",
#     "android.permission.SEND_SMS",
#     "android.permission.SET_ALARM",
#     "android.permission.SET_ALWAYS_FINISH",
#     "android.permission.SET_ANIMATION_SCALE",
#     "android.permission.SET_DEBUG_APP",
#     "android.permission.SET_PREFERRED_APPLICATIONS",
#     "android.permission.SET_PROCESS_LIMIT",
#     "android.permission.SET_TIME",
#     "android.permission.SET_TIME_ZONE",
#     "android.permission.SET_WALLPAPER",
#     "android.permission.SET_WALLPAPER_HINTS",
#     "android.permission.SIGNAL_PERSISTENT_PROCESSES",
#     "android.permission.STATUS_BAR",
#     "android.permission.SYSTEM_ALERT_WINDOW",
#     "android.permission.TRANSMIT_IR",
#     "android.permission.UNINSTALL_SHORTCUT",
#     "android.permission.UPDATE_DEVICE_STATS",
#     "android.permission.USE_BIOMETRIC",
#     "android.permission.USE_FINGERPRINT",
#     "android.permission.USE_SIP",
#     "android.permission.VIBRATE",
#     "android.permission.WAKE_LOCK",
#     "android.permission.WRITE_APN_SETTINGS",
#     "android.permission.WRITE_CALENDAR",
#     "android.permission.WRITE_CALL_LOG",
#     "android.permission.WRITE_CONTACTS",
#     "android.permission.WRITE_EXTERNAL_STORAGE",
#     "android.permission.WRITE_GSERVICES",
#     "android.permission.WRITE_SECURE_SETTINGS",
#     "android.permission.WRITE_SETTINGS",
#     "android.permission.WRITE_SYNC_SETTINGS",
#     "android.permission.WRITE_VOICEMAIL"
# ]


class SmaliFile(object):
    def __init__(self, dec_path, typ):
        super(SmaliFile, self).__init__()
        self.smali_folder_path = dec_path + "/smali"
        self.res_folder_path = dec_path + "/res"
        #self.smali_files, self.library_feature = self.getSmaliFiles(typ)
        self.smali_files = self.getCustomSmaliFiles(typ)
        self.library_feature = self.getLibraryFeature()
        a=5

    def getLibrary(self):
        return self.library_feature

    def getPackageName(self, dec_path):
        tree = etree.ElementTree(file="%s/AndroidManifest.xml" % dec_path)
        root = tree.getroot()
        packageName = root.attrib['package'].replace('.', '/')
        return packageName

    def getCustomSmaliFiles(self, typ):
            smali_files = []
            library_feature = {}
            packageName = self.getPackageName(os.path.dirname(self.smali_folder_path))

            for root, subs, files in os.walk("%s" % self.smali_folder_path):
                for filename in files:
                    if int(typ) == 0:
                        if packageName in root:
                            smali_files.append(os.path.join(root, filename))
                        #else:
                        #    library_feature[root.split('/smali/')[-1]] = 1
                            #print(root)
                    else:
                        smali_files.append(os.path.join(root, filename))
            return smali_files

    def getLibraryFeature(self):
            library_feature = {}
            libraries = []

            with open(os.getcwd() + "/Data/library/library_list".format(self.smali_folder_path.split('/')[4]), 'r+') as f:
                lineno = 0
                for line in f:
                    lineno += 1
                    libraries.append(line[:-1].strip())

            for lib in libraries:
                library_feature[lib] = 0

            for root, subs, files in os.walk("%s" % self.smali_folder_path):
                for lib in libraries:
                    if root.split('smali/')[-1].__eq__(lib):
                        library_feature[lib] = 1

            return library_feature

    # type 0 -> custom codes
    # type 1 -> all codes including library codes
    def getSmaliFiles(self, typ):
        smali_files = []
        libraries = []
        library_feature = {}


        with open(os.getcwd() + "/Data/library/library_list", 'r+') as f:
            lineno = 0
            for line in f:
                lineno += 1
                libraries.append(line[:-1])

        for lib in libraries:
            library_feature[lib] = 0

        if int(typ) == 0: #custom code
            # libs_name = lrd.tree.getLibs()
            for root, subs, files in os.walk(self.smali_folder_path):
                isLib = False
                #with open("/home/emreaydogan/Documents/lib2", 'a') as ff:
                #    if not "/data/Dataset/Smali" in root.split('smali/')[-1]:
                #        ff.write(root.split('smali/')[-1] + "\n")
                for lib in libraries:
                    #if lib in root.split('smali/')[-1]:
                    #if root.split('smali/')[-1] == "com/android":
                    #    aaaa = 5
                    if root.split('smali/')[-1].__eq__(lib):
                        isLib = True
                        library_feature[lib] = 1  ########################################### +1 de yapilabilir kac kere libraryi kullanmis
                        break
                if isLib == False:
                    for filename in files:
                        smali_files.append(os.path.join(root, filename))
            return smali_files, library_feature
        else:
            for root, subs, files in os.walk(self.smali_folder_path):
                for lib in libraries:
                    #if lib in root.split('smali')[-1]:
                    if root.split('smali/')[-1].__eq__(lib):
                        library_feature[lib] = 1
                        break
                for filename in files:
                    smali_files.append(os.path.join(root, filename))
            return smali_files, library_feature

    def getResFiles(self):
        res_files = []
        for root, subs, files in os.walk(self.res_folder_path):
            for filename in files:
                if filename.endswith(".xml"):
                    res_files.append(os.path.join(root, filename))
        return res_files

    def parseAppString(self):
        strings = []
        temp_regex = "/res/values.{0,}/strings.xml"
        res_files = self.getResFiles()
        strings_xml_files = [x for x in res_files if re.search(temp_regex, x)]

        for file in strings_xml_files:
            with open(file, "r") as f:
                for line in f:
                    match = re.search("<string name=", line)
                    if match:
                        s = re.search(">(.*)<", match.string)
                        if s:
                            strings.append(s.group(1))

        return strings

    def parsePermission(self, dec_path):
        tree = etree.ElementTree(file="%s/AndroidManifest.xml" % dec_path)
        root = tree.getroot()
        ns = root.nsmap['android']
        ns = "{%s}" % ns
        i = 0

        permission_feature = {}
        with open(os.getcwd() + "/Data/permission/permission_list_158", "r") as permission_file:
            for perm in permission_file:
                permission_feature[perm.strip()] = 0
            # permissions = permission_file.readlines()
            # for perm in permissions:
            #     permission_feature[perm.strip()] = 0

        for usedPer in root.iter("uses-permission"):
            perName = (usedPer.attrib[ns + 'name']).lower()
            if perName in permission_feature:
                permission_feature[perName] = 1  #####################permission_feature.keys() icinde var mi yok mu kontrol etmek lazim olabilir
                i += 1

        return permission_feature

    @property
    def parseSmali(self):
        smali_feature = {'avgCharPerLine': 0,
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
            for smali_file in self.smali_files:
                with open(smali_file, "r") as file:
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
                                    smali_feature["avgCharPerFuncName"] = float(
                                        smali_feature["avgCharPerFuncName"] * number_of_func + len(function_name)) / (number_of_func + 1)
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
                                    smali_feature["avgCharPerFuncParamName"] = float(smali_feature["avgCharPerFuncParamName"] * number_of_param + len(param_name)) / (number_of_param + 1)
                                    number_of_param += 1
                                continue
                            elif code.startswith(".field"):  # avg char per global var
                                glob = code.split(":")[0].split(" ")[-1]
                                smali_feature["avgCharPerGlobalVar"] = float(
                                    smali_feature["avgCharPerGlobalVar"] * number_of_global_var + len(glob)) / (number_of_global_var + 1)
                                if glob.isupper():
                                    number_of_upper_var += 1
                                number_of_global_var += 1
                                continue
                            elif code.startswith(".local"):  # avg char per local var
                                spl = code.split(",")
                                if len(spl) > 1:
                                    local = spl[1].split(":")[0][2:-1]
                                    smali_feature["avgCharPerLocalVar"] = float(
                                        smali_feature["avgCharPerLocalVar"] * number_of_local_var + len(local)) / (number_of_local_var + 1)
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
                smali_feature["avgCharPerLine"] = float(codes.__len__()) / number_of_line
                smali_feature["ratioGlobalVarToAllCodes"] = float(number_of_global_var) / number_of_line
                smali_feature["ratioLocalVarToAllCodes"] = float(number_of_local_var) / number_of_line
                smali_feature["ratioVarToAllCodes"] = float(number_of_local_var + number_of_global_var) / number_of_line
                smali_feature["ratioIfToAllCodes"] = float(number_of_if) / number_of_line
                smali_feature["ratioInvokeToAllCodes"] = float(number_of_invoke) / number_of_line
                smali_feature["ratioMoveToAllCodes"] = float(number_of_move) / number_of_line

            if number_of_func != 0:
                smali_feature["avgLinePerFunc"] = float(number_of_func_line) / number_of_func
                smali_feature["ratioIntToAllFunc"] = float(number_of_I_return_type) / number_of_func
                smali_feature["ratioVoidToAllFunc"] = float(number_of_V_return_type) / number_of_func
                smali_feature["ratioUpperToAllFunc"] = float(number_of_upper_func) / number_of_func

            if len(self.smali_files) != 0:
                smali_feature["avgLinePerClass"] = float(number_of_line) / len(self.smali_files)
                smali_feature["avgFuncPerClass"] = float(number_of_func) / len(self.smali_files)

            if (number_of_local_var + number_of_global_var) != 0:
                smali_feature["ratioUpperToAllVar"] = float(number_of_upper_var) / (number_of_local_var + number_of_global_var)


        except:
            err = str(sys.exc_info()[0]) + str(sys.exc_info()[1])
            print(err)

        return smali_feature
