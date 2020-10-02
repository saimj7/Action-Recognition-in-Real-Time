#=============================== \CONFIG./ =====================================
#===============================================================================
""" The following values must be same during training and inference """
TAKE_FRAME = 1
LOOK_BACK = 4
VGG16_OUT = 1024
SIZE = (224, 224)
#===============================================================================
""" Config. for inference source and alert feature """
# Webcam on/off: If off, source will be test video path by default.
FROM_WEBCAM = False
# Email alert on/off.
ALERT = False
# Enter the mail to receive alerts. E.g., 'xxx@gmail.com'.
MAIL = ''
# Optimize the threshold (avg. prediction score for class labels) if desired.
# 1 for class1 and 0 for class2. Currently set at 0.50 by default.
Threshold = 0.50
# Adjust the total_frames (avg. score to send the mail).
# Currently set to 5 by default.
positive_frames = 5
#===============================================================================
#===============================================================================
