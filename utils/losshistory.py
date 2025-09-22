import csv
import os

import scipy.signal
from matplotlib import pyplot as plt


class lossHistory:
    def __init__(self, save_dir, contra_type, train_type):
        self.save_dir = save_dir
        self.contra_type = contra_type
        self.train_type = train_type
        self.train_loss = []
        self.sup_loss = []
        self.mlm_loss = []
        self.self_loss = []
        self.ce_loss = []
        self.inter_loss = []
        self.type_loss = []
        self.cross_loss = []
        self.unlab_acc = [[], [], [], []]
        self.unlab_known_acc = [[], [], [], []]
        self.unlab_unknown_acc = [[], [], [], []]
        self.lab_acc = [[], [], [], []]
        self.best_k = [[], [], [], []]
        self.gold_k = None

        self.b3_prec = [[], [], [], []]
        self.b3_recall = [[], [], [], []]
        self.b3_f1 = [[], [], [], []]

        self.v_measure = [[], [], [], []]
        self.homogeneity = [[], [], [], []]
        self.completeness = [[], [], [], []]

        self.ari = [[], [], [], []]
        self.nmi = [[], [], [], []]

        self.val_acc = []
        self.val_b3_prec = []
        self.val_b3_recall = []
        self.val_b3_f1 = []
        self.val_vm = []
        self.val_homo = []
        self.val_complete = []
        self.val_ari = []
        self.val_nmi = []
        self.base2print = {0: 'lab', 1: 'unlab_tot', 2: 'unlab_known', 3: 'unlab_unknown'}
        self.sub_dirs = ['acc', 'b3', 'v_measure', 'ari', 'nmi', 'val', 'visual']
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for sub_dir in self.sub_dirs:
            dir_path = os.path.join(self.save_dir, sub_dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def record(self):
        with open(f'{self.save_dir}/info.csv', 'w', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['', 'lab acc', 'unlab tot acc', 'unlab known acc', 'unlab unknown acc', 'best k'])
            tmp = ['init lab', self.lab_acc[0][0], self.unlab_acc[0][0], self.unlab_known_acc[0][0],
                   self.unlab_unknown_acc[0][0], self.best_k[0][0]]
            w.writerow(tmp)
            for i in range(len(self.lab_acc)):
                if len(self.lab_acc[i]) > 0:
                    tmp = [self.base2print[i], self.lab_acc[i][-1], self.unlab_acc[i][-1], self.unlab_known_acc[i][-1],
                           self.unlab_unknown_acc[i][-1], self.best_k[i][-1]]
                    w.writerow(tmp)
            w.writerow([])
            w.writerow(['', 'b3 prec', 'b3 recall', 'b3 f1', 'v_measurre', 'homo', 'complete', 'ari', 'nmi'])
            for i in range(len(self.b3_prec)):
                if len(self.b3_prec[i]) > 0:
                    tmp = [f'init {self.base2print[i]}', self.b3_prec[i][0], self.b3_recall[i][0], self.b3_f1[i][0],
                           self.v_measure[i][0], self.homogeneity[i][0], self.completeness[i][0], self.ari[i][0],
                           self.nmi[i][0]]
                    w.writerow(tmp)
            for i in range(len(self.b3_prec)):
                if len(self.b3_prec[i]) > 0:
                    tmp = [self.base2print[i], self.b3_prec[i][-1], self.b3_recall[i][-1], self.b3_f1[i][-1],
                           self.v_measure[i][-1], self.homogeneity[i][-1], self.completeness[i][-1], self.ari[i][-1],
                           self.nmi[i][-1]]
                    w.writerow(tmp)

    def append_val(self, lab_acc, b3, vm_ari_nmi):
        self.val_acc.append(lab_acc)
        self.val_b3_prec.append(b3[0][0])
        self.val_b3_recall.append(b3[0][1])
        self.val_b3_f1.append(b3[0][2])
        self.val_vm.append(vm_ari_nmi[0][0])
        self.val_homo.append(vm_ari_nmi[0][1])
        self.val_complete.append(vm_ari_nmi[0][2])
        self.val_ari.append(vm_ari_nmi[0][3])
        self.val_nmi.append(vm_ari_nmi[0][4])

        with open(os.path.join(self.save_dir, f"val/log.txt"), 'a') as f:
            f.write(f'val acc: {lab_acc};b3 prec: {b3[0][0]}; b3 recall: {b3[0][1]}; b3 f1: {b3[0][2]}'
                    f'v_measure: {vm_ari_nmi[0][0]}; homogeneity: {vm_ari_nmi[0][1]}; '
                    f'completeness: {vm_ari_nmi[0][2]}; ari: {vm_ari_nmi[0][3]}; nmi: {vm_ari_nmi[0][4]}\n')

    def append_loss(self, tl=0, sul=0, mcl=0, sel=0, inter=0, typel=0, cl=0):
        if self.train_type == 'pretrain':
            if self.contra_type == 'sup':
                self.train_loss.append(tl)
                self.sup_loss.append(sul)
                self.mlm_loss.append(mcl)
                self.ce_loss.append(sel)
                with open(os.path.join(self.save_dir, f"loss.txt"), 'a') as f:
                    f.write(f'tot loss {tl};\tsup loss {sul};\tmlm loss {mcl};\tce loss {sel}.\n')
            elif self.contra_type in ['self', 'test']:
                self.train_loss.append(tl)
                self.sup_loss.append(sul)
                self.ce_loss.append(mcl)
                self.self_loss.append(sel)
                with open(os.path.join(self.save_dir, f"loss.txt"), 'a') as f:
                    f.write(f'tot loss {tl};\tsup loss {sul};\tce loss {mcl};\tself loss {sel}.\n')
        elif self.train_type == 'train':
            self.train_loss.append(tl)
            self.sup_loss.append(sul)
            self.ce_loss.append(mcl)
            self.self_loss.append(sel)
            self.inter_loss.append(inter)
            self.type_loss.append(typel)
            self.cross_loss.append(cl)
            with open(os.path.join(self.save_dir, f"loss.txt"), 'a') as f:
                f.write(f'tot loss {tl};\tsup loss {sul};\tpcl loss {mcl};\tre loss {sel};\tinter loss {inter};'
                        f'\ttype loss {typel};\tcross loss {cl}.\n')

    def append_eval(self, cluster_acc, b3, vm_ari_nmi):
        # 0 lab 1 unlab 2 unlab_known 3 unlab_unknown
        for i in range(len(cluster_acc)):
            self.unlab_acc[i].append(cluster_acc[i][0])
            self.unlab_known_acc[i].append(cluster_acc[i][1])
            self.unlab_unknown_acc[i].append(cluster_acc[i][2])
            self.lab_acc[i].append(cluster_acc[i][3])
            self.best_k[i].append(cluster_acc[i][4])

            with open(os.path.join(self.save_dir, f"acc/log_acc_{self.base2print[i]}.txt"), 'a') as f:
                f.write(f'unlab_tot_acc: {cluster_acc[i][0]};\t'
                        f'unlab_known_acc: {cluster_acc[i][1]};\t'
                        f'unlab_unknown_acc: {cluster_acc[i][2]};\t'
                        f'lab_acc: {cluster_acc[i][3]};\t'
                        f'besk_k: {cluster_acc[i][4]};\t')
                # ttt = ''
                # for idx, j in enumerate(cluster_acc[i][5]):
                #     ttt += f'level {idx} k: {j};'
                # f.write(ttt)
                f.write('\n')

        for i in range(len(b3)):
            self.b3_prec[i].append(b3[i][0])
            self.b3_recall[i].append(b3[i][1])
            self.b3_f1[i].append(b3[i][2])

            with open(os.path.join(self.save_dir, f"b3/log_b3_{self.base2print[i]}.txt"), 'a') as f:
                f.write(f'b3_prec: {b3[i][0]};\tb3_recall: {b3[i][1]};\tb3_f1: {b3[i][2]}.\n')

        for i in range(len(vm_ari_nmi)):
            self.v_measure[i].append(vm_ari_nmi[i][0])
            self.homogeneity[i].append(vm_ari_nmi[i][1])
            self.completeness[i].append(vm_ari_nmi[i][2])
            self.ari[i].append(vm_ari_nmi[i][3])
            self.nmi[i].append(vm_ari_nmi[i][4])

            with open(os.path.join(self.save_dir, f"v_measure/log_v_measure_{self.base2print[i]}.txt"), 'a') as f:
                f.write(f'v_measure(f1): {vm_ari_nmi[i][0]};\thomogeneity: {vm_ari_nmi[i][1]};\t'
                        f'completeness: {vm_ari_nmi[i][2]}.\n')
            with open(os.path.join(self.save_dir, f"ari/log_ari_{self.base2print[i]}.txt"), 'a') as f:
                f.write(f'ari: {vm_ari_nmi[i][3]}.\n')
            with open(os.path.join(self.save_dir, f"nmi/log_nmi_{self.base2print[i]}.txt"), 'a') as f:
                f.write(f'nmi: {vm_ari_nmi[i][4]}.\n')

    def _plot_metric(self, iters, y_values, labels, colors, title, ylabel, filename):
        """
        辅助函数：绘制折线图并保存
        """
        plt.figure()
        for y, label, color in zip(y_values, labels, colors):
            plt.plot(iters, y, color, linewidth=2, label=label)
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.title(title)
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.cla()
        plt.close("all")

    def loss_plot(self):
        iters = range(len(self.train_loss))

        if self.train_type == 'pretrain':
            if self.contra_type == 'sup':
                dd = [self.train_loss, self.sup_loss, self.mlm_loss, self.ce_loss]
                tt = ['train loss', 'sup loss', 'mlm loss', 'ce loss']
                cc = ['red', 'blue', 'green', 'purple']
            elif self.contra_type in ['self', 'test']:
                dd = [self.train_loss, self.sup_loss, self.ce_loss, self.self_loss]
                tt = ['train loss', 'sup loss', 'ce loss', 'self loss']
                cc = ['red', 'blue', 'green', 'purple']
        elif self.train_type == 'train':
            dd = [self.train_loss, self.sup_loss, self.ce_loss, self.self_loss, self.inter_loss, self.type_loss, self.cross_loss]
            tt = ['train loss', 'sup loss', 'pcl loss', 're loss', 'inter loss', 'type loss', 'cross loss']
            cc = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink']
        else:
            tt = []
            dd = []
            cc = []
        self._plot_metric(
            iters,
            dd,
            tt,
            cc,
            'Training Loss',
            'Loss',
            "epoch_loss.png"
        )

        iters = range(len(self.lab_acc[0]))
        for i in range(len(self.base2print)):
            if len(self.unlab_acc[i]) > 0:
                self._plot_metric(
                    iters,
                    [self.unlab_acc[i], self.unlab_known_acc[i], self.unlab_unknown_acc[i], self.lab_acc[i]],
                    ['unlab acc', 'unlab known acc', 'unlab unknown acc', 'lab acc'],
                    ['red', 'blue', 'green', 'purple'],
                    'Accuracy',
                    'Acc',
                    f"acc/epoch_acc_{self.base2print[i]}.png"
                )

            if len(self.best_k[i]) > 0:
                plt.figure()
                plt.plot(iters, self.best_k[i], 'red', linewidth=2, label='best k')
                if self.gold_k is not None:
                    plt.axhline(y=self.gold_k, color='blue', linestyle='--', linewidth=2, label='target k')
                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('best_k')
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(self.save_dir, f"acc/epoch_k_{self.base2print[i]}.png"))
                plt.cla()
                plt.close("all")

            if len(self.b3_prec[i]) > 0:
                self._plot_metric(
                    iters,
                    [self.b3_prec[i], self.b3_recall[i], self.b3_f1[i]],
                    ['b3 prec', 'b3 recall', 'b3 f1'],
                    ['red', 'blue', 'green'],
                    'B3 Scores',
                    'B3',
                    f"b3/epoch_b3_{self.base2print[i]}.png"
                )

            if len(self.v_measure[i]) > 0:
                self._plot_metric(
                    iters,
                    [self.v_measure[i], self.homogeneity[i], self.completeness[i]],
                    ['v_measure(f1)', 'homogeneity', 'completeness'],
                    ['red', 'blue', 'green'],
                    'V-measure Scores',
                    'V-measure',
                    f"v_measure/epoch_v_measure_{self.base2print[i]}.png"
                )

            if len(self.ari[i]) > 0:
                self._plot_metric(
                    iters,
                    [self.ari[i]],
                    ['ari'],
                    ['red'],
                    'Adjusted Rand Index (ARI)',
                    'ARI',
                    f"ari/epoch_ari_{self.base2print[i]}.png"
                )

            if len(self.nmi[i]) > 0:
                self._plot_metric(
                    iters,
                    [self.nmi[i]],
                    ['nmi'],
                    ['red'],
                    'NMI',
                    'NMI',
                    f"nmi/epoch_nmi_{self.base2print[i]}.png"
                )

        iters = range(len(self.val_acc))
        if len(self.val_acc) > 0:
            self._plot_metric(
                iters,
                [self.val_acc],
                ['lab acc'],
                ['red'],
                'Accuracy',
                'Acc',
                f"val/epoch_acc.png"
            )
            self._plot_metric(
                iters,
                [self.val_b3_prec, self.val_b3_recall, self.val_b3_f1],
                ['b3 prec', 'b3 recall', 'b3 f1'],
                ['red', 'blue', 'green'],
                'B3 Scores',
                'B3',
                f"val/epoch_b3.png"
            )
            self._plot_metric(
                iters,
                [self.val_vm, self.val_homo, self.val_complete],
                ['v_measure(f1)', 'homogeneity', 'completeness'],
                ['red', 'blue', 'green'],
                'V-measure Scores',
                'V-measure',
                f"val/epoch_v_measure.png"
            )
            self._plot_metric(
                iters,
                [self.val_ari],
                ['ari'],
                ['red'],
                'Adjusted Rand Index (ARI)',
                'ARI',
                f"val/epoch_ari.png"
            )
            self._plot_metric(
                iters,
                [self.val_nmi],
                ['nmi'],
                ['red'],
                'NMI',
                'NMI',
                f"val/epoch_nmi.png"
            )
