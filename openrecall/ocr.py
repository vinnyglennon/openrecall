_ocr_model = None


def _get_ocr_model():
    """Lazily load the OCR predictor to avoid import-time cost."""
    global _ocr_model
    if _ocr_model is None:
        from doctr.models import ocr_predictor

        _ocr_model = ocr_predictor(
            pretrained=True,
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_mobilenet_v3_large",
        )
    return _ocr_model


def extract_text_from_image(image):
    ocr = _get_ocr_model()
    result = ocr([image])
    text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text += word.value + " "
                text += "\n"
            text += "\n"
    return text
