from xgutils import omgutil

example_omage_1024 = "assets/B0742FHDJF_objamage_tensor_1024.npz"
# example_omage_1024 = 'test.npy'
omage = omgutil.load_omg(example_omage_1024)
vomg, rdimg, _ = omgutil.preview_omg(omage)
visutil.showImg(vomg)
