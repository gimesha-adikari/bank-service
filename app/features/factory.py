from app.core.config import get_settings
from app.features.face.simple import SimpleFaceMatcher
from app.features.liveness.simple import SimpleLivenessDetector
from app.features.ocr.simple import SimpleOcrEngine
from app.features.doccls.simple import SimpleDocClassifier
from app.features.portrait.simple import SimplePortraitExtractor


class FeatureFactory:
    def __init__(self):
        self.settings = get_settings()
        # For now only 'simple'. Later: branch on settings.face_backend == 'insightface', etc.
        self._face = SimpleFaceMatcher()
        self._live = SimpleLivenessDetector()
        self._ocr  = SimpleOcrEngine()
        self._doc  = SimpleDocClassifier()
        self._portrait = SimplePortraitExtractor() if self.settings.portrait_backend != "off" else None

    def face_matcher(self):   return self._face
    def liveness(self):       return self._live
    def ocr(self):            return self._ocr
    def doc_classifier(self): return self._doc
    def portrait(self):       return self._portrait

_factory = None
def get_factory() -> FeatureFactory:
    global _factory
    if _factory is None:
        _factory = FeatureFactory()
    return _factory
