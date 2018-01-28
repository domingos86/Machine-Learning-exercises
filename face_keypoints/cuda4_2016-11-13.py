# IPython log file

import kfkd_single_model as kfkd
net = kfkd.fit()
kfkd.plot_net(net)
kfkd.plot_net(net)
kfkd.predict(net, save_to='submission2.csv')
kfkd.save_net('single_model.pickle', net)
get_ipython().magic(u'pinfo kfkd.save_net')
kfkd.save_net(net,'single_model.pickle')
exit()
