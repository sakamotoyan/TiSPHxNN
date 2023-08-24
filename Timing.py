import warnings
import time

class Timing:
    def __init__(self, filename = ""):
        self._filename = filename
        self._stepId = 0
        self._stepLength = -1.0
        self._groups = []
        self._groupTime = {} # key: group name; val: time consumption this step
        self._group = None # current group name
        self._time = 0.0

    def addGroup(self, groupName):
        if groupName in self._groups:
            warnings.warn("Group " + groupName + " already exists, will not be added again.", UserWarning)
            return
        if self._stepId != 0:
            warnings.warn("Step already started, can't add more groups.", UserWarning)
            return
        self._groups.append(groupName)
        self._groupTime[groupName] = 0.0

    def startStep(self):
        self._stepId += 1
        self._stepLength = -1.0
        for g in self._groupTime:
            self._groupTime[g] = 0.0

    def setStepLength(self, length):
        self._stepLength = length
    
    def endStep(self):
        if not self._group is None:
            warnings.warn("Did not end group: " + self._group + ", will automatically end it.", UserWarning)
            self.endGroup()
        l = [self._stepId, self._stepLength]
        for group in self._groups:
            l.append(self._groupTime[group])
        self._outputStep(l)

    def startGroup(self, groupName):
        if not self._group is None:
            warnings.warn("Did not end group: " + self._group + ", will automatically end it.", UserWarning)
            self.endGroup()
        if groupName in self._groupTime:
            self._group = groupName
            self._time = time.time()
        else:
            warnings.warn("Group doesn't exist: " + groupName + ", will not record.", UserWarning)


    def endGroup(self):
        if self._group is None:
            warnings.warn("endGroup called before startGroup.", UserWarning)
            return
        self._groupTime[self._group] += time.time() - self._time
        self._group = None

    def _outputGroups(self, colSeparator = ',', rowSeparator = '\n'):
        with open(self._filename, 'w') as file:
            file.write("id")
            file.write(colSeparator)
            file.write("timestep")
            for i in range(len(self._groups)):
                file.write(colSeparator)
                file.write(self._groups[i])
            file.write(rowSeparator)

    def _outputStep(self, stepData, colSeparator = ',', rowSeparator = '\n'):
        stepId = self._stepId
        if stepId == 1:
            self._outputGroups()
        with open(self._filename, 'a') as file:
            for i in range(len(stepData) - 1):
                file.write(str(stepData[i]))
                file.write(colSeparator)
            file.write(str(stepData[len(stepData) - 1]))
            file.write(rowSeparator)
