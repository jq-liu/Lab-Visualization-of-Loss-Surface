def project_trajectory(dir_file, w, s, dataset, model_name, model_files, dir_type='weights', proj_method='cos'):

    proj_file = dir_file + '_proj_' + proj_method + '.h5'

    # read directions and convert them to vectors
    directions = net_plotter.load_directions(dir_file)
    dx = nplist_to_tensor(directions[0])
    dy = nplist_to_tensor(directions[1])

    xcoord, ycoord = [], []
    for model_file in model_files:
        net2 = model_loader.load(dataset, model_name, model_file)
        if dir_type == 'weights':
            w2 = net_plotter.get_weights(net2)
            d = net_plotter.get_diff_weights(w, w2)
        elif dir_type == 'states':
            s2 = net2.state_dict()
            d = net_plotter.get_diff_states(s, s2)
        d = tensorlist_to_tensor(d)

        x, y = project_2D(d, dx, dy, proj_method)
        print ("%s  (%.4f, %.4f)" % (model_file, x, y))

        xcoord.append(x)
        ycoord.append(y)

    f = h5py.File(proj_file, 'w')
    f['proj_xcoord'] = np.array(xcoord)
    f['proj_ycoord'] = np.array(ycoord)
    f.close()

    return proj_file

def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    xdirection = h5_util.read_list(f, 'xdirection')
    ydirection = h5_util.read_list(f, 'ydirection')
    directions = [xdirection, ydirection]

    return directions

def nplist_to_tensor(nplist):
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)
