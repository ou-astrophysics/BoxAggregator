class SaveableStore:

    @classmethod
    def getSaveableAttrs(cls):
        return cls.saveableAttrs

    @classmethod
    def fromSaveable(cls, saveable):
        new = cls()
        for saveableAttr in cls.getSaveableAttrs():
            setattr(new, saveableAttr, saveable[saveableAttr])
        return new

    def getSaveable(self):
        return {
            saveableAttr: getattr(self, saveableAttr, None)
            for saveableAttr in self.getSaveableAttrs()
        }
