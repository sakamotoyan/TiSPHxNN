import warnings

class Statistics:
    def __init__(self, filename = ""):
        self._filename = filename
        self._stepId = 0
        self._groups = []

    def addGroup(self, groupName):
        if groupName in self._groups:
            warnings.warn("Group " + groupName + " already exists, will not be added again.", UserWarning)
            return
        if self._stepId != 0:
            warnings.warn("Step already started, can't add more groups.", UserWarning)
            return
        self._groups.append(groupName)

    def recordStep(self, simTime = -1, **kwargs):
        self._stepId += 1
        l = [self._stepId, simTime]
        for group in self._groups:
            if group in kwargs:
                l.append(kwargs[group])
            else:
                l.append("")
        self._outputStep(l)


    def _outputGroups(self, colSeparator = ',', rowSeparator = '\n'):
        with open(self._filename, 'w') as file:
            file.write("id")
            file.write(colSeparator)
            file.write("sim_time")
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

if __name__ == "__main__":
    s = Statistics("testStatistics.csv")
    s.addGroup("A")
    s.addGroup("B")
    s.recordStep(A=1)
    s.recordStep(1.5,B=2)
    s.recordStep(2.0,A=3,B=4)