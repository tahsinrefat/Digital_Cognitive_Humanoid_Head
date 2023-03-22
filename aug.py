import Augmentor 
p = Augmentor.Pipeline(r"/home/mouneeta/Desktop/facenet/face recognition/train_img/amrin")

p.random_brightness(probability=0.3,min_factor=0.3, max_factor=1.2)
p.random_distortion(probability=1, grid_width=2,grid_height=2, magnitude=3)
p.sample(100)

q = Augmentor.Pipeline(r"/home/mouneeta/Desktop/facenet/face recognition/train_img/mouneeta")

q.random_brightness(probability=0.3,min_factor=0.3, max_factor=1.2)
q.random_distortion(probability=1, grid_width=2,grid_height=2, magnitude=3)
q.sample(100)

t = Augmentor.Pipeline(r"/home/mouneeta/Desktop/facenet/face recognition/train_img/pinky")

t.random_brightness(probability=0.3,min_factor=0.3, max_factor=1.2)
t.random_distortion(probability=1, grid_width=2,grid_height=2, magnitude=3)
t.sample(100)





