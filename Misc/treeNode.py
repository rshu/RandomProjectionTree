class treeNode(object):
    def __init__(self, name, currentDepth, parent = None, leftItems=[], rightItems=[]):
        self.name = name
        self.currentDepth = currentDepth
        self.parent = parent

        if leftItems:
            self.leftItems = leftItems.copy()
        if rightItems:
            self.rightItems = rightItems.copy()
