from model.compare_fusion_model_1 import MutualCrossAttention
from model.compare_fusion_model_2 import FusionModel2
from model.compare_fusion_model_4 import FusionModel4
from model.compare_fusion_model_5 import FusionModel5
from model.image_fusion import *
from model.pps_cnn import *




class ModalFusionEncoder(nn.Module):
    def __init__(self):
        super(ModalFusionEncoder,self).__init__()

        self.image_model = pyramid_trans_expr()
        self.pps_model = cnn_classifier()

        # self.fusion_model = MultimodalTransformer()
        self.fusion_model = FusionModel2()

    def forward(self, x_ir, x_face, pps_x):
        image_fusion_x = self.image_model(x_ir, x_face)
        pps_fusion_x = self.pps_model(pps_x)

        out, fusion_feature = self.fusion_model(image_fusion_x, pps_fusion_x)
        # out, x_a1, x_a2, x_b1, x_b2 = self.fusion_model(image_fusion_x, pps_fusion_x)

        return out, fusion_feature
        # return out, image_fusion_x, pps_fusion_x, x_a1, x_a2, x_b1, x_b2