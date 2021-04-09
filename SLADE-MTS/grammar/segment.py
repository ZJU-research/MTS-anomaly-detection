class SegmentIndex(object):

    def __init__(self, indices):
        self.indices = indices
        self.segments = []

    def printContent(self):
        print(str(self.indices[0]) + '---' + str(self.indices[1]))
        #print(id(self.segments))
        for segment in self.segments:
            segment.printLetters()

    def addSegment(self, segment):
        self.segments.append(segment)

    def getSegment(self, index):
        return self.segments[index]

class Segment(object):

    def __init__(self, series, letters, indices, segmentIndex):
        self.series = series
        self.letters = letters
        self.indices = indices
        self.segmentIndex = segmentIndex
        self.coverdRules = None
        self.allCoveredRules = None
    
    def printLetters(self):
        print(self.letters)
    
    def content(self):
        content = self.letters+str(self.segmentIndex.indices[1])
        return content

    def setRule(self, rule):
        self.coverdRules = rule
    
    def getRule(self):
        return self.coverdRules
