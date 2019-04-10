class Face():
    def __init__(self):
        import os
        before_folder = os.path.abspath('.')
        os.chdir(
            os.path.join('/'.join(Face.__module__.split('.')[:-1]), 'DFace'))
        from dface.core.detect import create_mtcnn_net, MtcnnDetector
        pnet, rnet, onet = create_mtcnn_net(
            p_model_path="./model_store/pnet_epoch.pt",
            r_model_path="./model_store/rnet_epoch.pt",
            o_model_path="./model_store/onet_epoch.pt",
            use_cuda=False)
        self.detector = MtcnnDetector(
            pnet=pnet, rnet=rnet, onet=onet, min_face_size=64)
        os.chdir(before_folder)

    def get_face_from_file(self, org_file, margin=5.):
        import imageio
        img = imageio.imread(org_file, pilmode="RGB")
        try:
            bbox = map(int,
                       self.detector.detect_face(img[:, :, ::-1])[0][0][:-1])
            bbox = [max(0, i) for i in bbox]

            def marginP(x, y):
                return int(x + y / margin)

            def marginM(x, y):
                return int(x - y / margin)

            bbox = [
                marginM(bbox[0], bbox[2] - bbox[0]),
                marginM(bbox[1], bbox[3] - bbox[1]),
                marginP(bbox[2], bbox[2] - bbox[0]),
                marginP(bbox[3], bbox[3] - bbox[1])
            ]
            bbox = [max(0, i) for i in bbox]
            img_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            return img_face, True
        except IndexError:
            img_face = img
            return img_face, False

    def get_all_faces_from_file(self, org_file, margin=5.):
        import imageio
        img = imageio.imread(org_file, pilmode="RGB")
        bboxes = [
            _bbox[:-1]
            for _bbox in self.detector.detect_face(img[:, :, ::-1])[0]
        ]
        good_bboxes = []
        for n in range(len(bboxes)):
            try:
                bbox = map(int, bboxes[n])
                bbox = [max(0, i) for i in bbox]

                def marginP(x, y):
                    return int(x + y / margin)

                def marginM(x, y):
                    return int(x - y / margin)

                bbox = [
                    marginM(bbox[0], bbox[2] - bbox[0]),
                    marginM(bbox[1], bbox[3] - bbox[1]),
                    marginP(bbox[2], bbox[2] - bbox[0]),
                    marginP(bbox[3], bbox[3] - bbox[1])
                ]
                bbox = [max(0, i) for i in bbox]
                # img_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                good_bboxes.append(bbox)
            except IndexError:
                continue
        return good_bboxes

    def get_face_and_save(self, org_file, face_file):
        import imageio
        import os
        if not os.path.isfile(face_file):
            face, success = self.get_face_from_file(org_file)
            imageio.imwrite(face_file, face)
            return success
        else:
            return False
