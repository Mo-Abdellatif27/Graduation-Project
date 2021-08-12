import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon, QMovie
import pandas as pd
from PandasModel import PandasModel
from My_Prediction import Prediction


class Ui_Dialog(object):
    def __init__(self):
        self.tr_path = 'no Info'
        self.ts_path = 'no Info'
        self.out_path = 'no Info'

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Price recommender")
        Dialog.resize(606, 508)
        self.tr_lineEdit = QtWidgets.QLineEdit(Dialog)
        self.tr_lineEdit.setGeometry(QtCore.QRect(100, 20, 431, 20))
        self.tr_lineEdit.setObjectName("tr_lineEdit")
        self.ts_lineEdit = QtWidgets.QLineEdit(Dialog)
        self.ts_lineEdit.setGeometry(QtCore.QRect(100, 60, 431, 20))
        self.ts_lineEdit.setObjectName("ts_lineEdit")
        self.tr_label = QtWidgets.QLabel(Dialog)
        self.tr_label.setGeometry(QtCore.QRect(20, 20, 71, 21))
        self.tr_label.setObjectName("tr_label")
        self.ts_label = QtWidgets.QLabel(Dialog)
        self.ts_label.setGeometry(QtCore.QRect(20, 60, 71, 21))
        self.ts_label.setObjectName("ts_label")

        self.pushButton_tr = QtWidgets.QPushButton(Dialog)
        self.pushButton_tr.setGeometry(QtCore.QRect(540, 20, 51, 23))
        self.pushButton_tr.setObjectName("pushButton_tr")
        self.pushButton_ts = QtWidgets.QPushButton(Dialog)
        self.pushButton_ts.setGeometry(QtCore.QRect(540, 60, 51, 23))
        self.pushButton_ts.setObjectName("pushButton_ts")
        self.pushButton_out = QtWidgets.QPushButton(Dialog)
        self.pushButton_out.setGeometry(QtCore.QRect(540, 100, 51, 23))
        self.pushButton_out.setObjectName("pushButton_out")

        self.out_lineEdit = QtWidgets.QLineEdit(Dialog)
        self.out_lineEdit.setGeometry(QtCore.QRect(100, 100, 431, 20))
        self.out_lineEdit.setObjectName("out_lineEdit")
        self.out_label = QtWidgets.QLabel(Dialog)
        self.out_label.setGeometry(QtCore.QRect(20, 100, 71, 21))
        self.out_label.setObjectName("out_label")

        #####################
        self.ts_label_2 = QtWidgets.QLabel(Dialog)
        self.ts_label_2.setGeometry(QtCore.QRect(20, 100, 71, 21))
        self.ts_label_2.setObjectName("ts_label_2")
        #####################

        self.pushButton_Run = QtWidgets.QPushButton(Dialog)
        self.pushButton_Run.setGeometry(QtCore.QRect(520, 140, 75, 23))
        self.pushButton_Run.setObjectName("pushButton_Run")

        self.pushButton_Run_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_Run_2.setGeometry(QtCore.QRect(20, 250, 101, 23))
        self.pushButton_Run_2.setObjectName("pushButton_Run_2")

        self.pushButton_Run_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_Run_3.setGeometry(QtCore.QRect(472, 250, 120, 23))
        self.pushButton_Run_3.setObjectName("pushButton_Run_3")

        self.tableView = QtWidgets.QTableView(Dialog)
        self.tableView.setGeometry(QtCore.QRect(20, 290, 571, 211))
        self.tableView.setObjectName("tableView")

        #####################
        self.comboBox_ct1 = QtWidgets.QComboBox(Dialog)
        self.comboBox_ct1.setGeometry(QtCore.QRect(20, 210, 101, 22))
        self.comboBox_ct1.setObjectName("comboBox_ct1")
        self.comboBox_ct2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_ct2.setGeometry(QtCore.QRect(130, 210, 101, 22))
        self.comboBox_ct2.setObjectName("comboBox_ct2")
        self.comboBox_ct3 = QtWidgets.QComboBox(Dialog)
        self.comboBox_ct3.setGeometry(QtCore.QRect(240, 210, 101, 22))
        self.comboBox_ct3.setObjectName("comboBox_ct3")
        self.comboBox_brand = QtWidgets.QComboBox(Dialog)
        self.comboBox_brand.setGeometry(QtCore.QRect(350, 210, 101, 22))
        self.comboBox_brand.setObjectName("comboBox_brand")
        self.comboBox_ship = QtWidgets.QComboBox(Dialog)
        self.comboBox_ship.setGeometry(QtCore.QRect(460, 210, 61, 22))
        self.comboBox_ship.setObjectName("comboBox_ship")
        self.comboBox_cond = QtWidgets.QComboBox(Dialog)
        self.comboBox_cond.setGeometry(QtCore.QRect(530, 210, 61, 22))
        self.comboBox_cond.setObjectName("comboBox_cond")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(140, 250, 81, 16))
        self.label.setObjectName("label")

        self.label_p_price = QtWidgets.QLabel(Dialog)
        self.label_p_price.setGeometry(QtCore.QRect(230, 250, 91, 16))
        self.label_p_price.setObjectName("label_p_price")

        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 190, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(130, 190, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(240, 190, 47, 13))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(350, 190, 61, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(460, 190, 51, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(530, 190, 51, 16))
        self.label_8.setObjectName("label_8")

        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(20, 160, 571, 20))
        self.label_10.setObjectName("label_10")
        #####################

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.tr_lineEdit.setText(_translate("Dialog", "no info"))
        self.ts_lineEdit.setText(_translate("Dialog", "no info"))
        self.out_lineEdit.setText(_translate("Dialog", "no info"))

        self.tr_label.setText(_translate("Dialog", "Train Dataset"))
        self.ts_label.setText(_translate("Dialog", "Test Dataset"))
        self.out_label.setText(_translate("Dialog", "Output to folder"))

        self.pushButton_tr.setText(_translate("Dialog", "Browse"))
        self.pushButton_ts.setText(_translate("Dialog", "Browse"))
        self.pushButton_out.setText(_translate("Dialog", "Browse"))
        self.pushButton_Run.setText(_translate("Dialog", "Start Train"))
        self.pushButton_Run_2.setText(_translate("Dialog", "Predicted value"))
        self.pushButton_Run_3.setText(_translate("Dialog", "Predicted new test file"))

        self.pushButton_tr.clicked.connect(self.pushButton_tr_handler)
        self.pushButton_ts.clicked.connect(self.pushButton_ts_handler)
        self.pushButton_out.clicked.connect(self.pushButton_out_handler)
        self.pushButton_Run.clicked.connect(self.start_new_predicted)
        self.pushButton_Run_2.clicked.connect(self.start_predicted_value)
        self.pushButton_Run_3.clicked.connect(self.start_predicted_df)

        self.updateCat1()
        self.comboBox_ship.addItem('_')
        self.comboBox_ship.addItem('0')
        self.comboBox_ship.addItem('1')

        self.comboBox_cond.addItem('_')
        for i in range(1, 6):
            self.comboBox_cond.addItem(str(i))

        self.comboBox_ct1.currentIndexChanged.connect(self.updateCat2)
        self.comboBox_ct2.currentIndexChanged.connect(self.updateCat3)
        self.comboBox_ct3.currentIndexChanged.connect(self.updateBrand)

        #####################
        self.label.setText(_translate("Dialog", "Predicted Price:"))
        self.label_p_price.setText(_translate("Dialog", "0"))
        self.label_3.setText(_translate("Dialog", "CAT1"))
        self.label_4.setText(_translate("Dialog", "CAT2"))
        self.label_5.setText(_translate("Dialog", "CAT3"))
        self.label_6.setText(_translate("Dialog", "Brand Name"))
        self.label_7.setText(_translate("Dialog", "Shipping"))
        self.label_8.setText(_translate("Dialog", "condition"))
        self.label_10.setText(_translate("Dialog",
                                         "______________________________________________________________________________________________________________________________________________________________________________________________"))
        #####################

    def pushButton_tr_handler(self):
        print("Train Browse Button pressed")
        filename = QFileDialog.getOpenFileName(filter=("Image Files (*.csv *.tsv *.xtxx)"))
        self.tr_path = filename[0]
        print(self.tr_path)
        print(filename[1])
        self.tr_lineEdit.setText(self.tr_path)

    def pushButton_ts_handler(self):
        print("Test Browse Button pressed")
        filename = QFileDialog.getOpenFileName(filter=("Image Files (*.csv *.tsv *.xtxx)"))
        self.ts_path = filename[0]
        print(self.ts_path)
        self.ts_lineEdit.setText(self.ts_path)

    def pushButton_out_handler(self):
        print("out Browse Button pressed")
        filename = QFileDialog.getExistingDirectory()
        self.out_path = filename
        print(self.out_path)
        self.out_lineEdit.setText(self.out_path)

    def loadFile_out(self, file_name):
        df = pd.read_csv(self.out_path+file_name)
        self.load_df_to_table(df)

    def load_df_to_table(self, df: pd.DataFrame):
        model = PandasModel(df)
        self.tableView.setModel(model)

    def updateCat1(self):
        self.comboBox_ct1.clear()
        self.comboBox_ct1.addItem('_')
        self.cat_df = pd.read_csv(".\\cat_table\\cat_df.csv")
        for i in set(list(self.cat_df['cat1'])):
            self.comboBox_ct1.addItem(i)

    def updateCat2(self):
        ct1 = str(self.comboBox_ct1.currentText())
        if ct1 != '_':
            self.comboBox_ct2.clear()
            self.comboBox_ct2.addItem('_')
            for i in set(list(self.cat_df[self.cat_df['cat1'] == ct1]['cat2'])):
                self.comboBox_ct2.addItem(i)

    def updateCat3(self):
        ct1 = str(self.comboBox_ct1.currentText())
        ct2 = str(self.comboBox_ct2.currentText())
        if ct1 != '_' and ct2 != '_':
            self.comboBox_ct3.clear()
            self.comboBox_ct3.addItem('_')
            ct2_df = self.cat_df[self.cat_df['cat1'] == ct1]
            for i in set(list(ct2_df[ct2_df['cat2'] == ct2]['cat3'])):
                self.comboBox_ct3.addItem(i)

    def updateBrand(self):
        ct1 = str(self.comboBox_ct1.currentText())
        ct2 = str(self.comboBox_ct2.currentText())
        ct3 = str(self.comboBox_ct3.currentText())
        if ct1 != '_' and ct2 != '_' and ct3 != '_':
            self.comboBox_brand.clear()
            self.comboBox_brand.addItem('_')
            ct2_df = self.cat_df[self.cat_df['cat1'] == ct1]
            ct3_df = ct2_df[ct2_df['cat2'] == ct2]
            for i in set(list(ct3_df[ct3_df['cat3'] == ct3]['brand_name'])):
                self.comboBox_brand.addItem(i)

    def start_new_predicted(self):
        predicted_thread = threading.Thread(target=self.new_predicted)
        predicted_thread.start()

    def new_predicted(self):
        self.app_btn_disable()
        p = Prediction()
        try:
            df_train = pd.read_csv(self.tr_path)
            df_test = pd.read_csv(self.ts_path)
        except:
            print('File read Error')
            self.app_btn_enable()
            return

        train_1 = p.tidy_up_train_data(df_train)
        test_1 = p.tidy_up_test_data(df_test, train_1)
        result = p.prediction(train_1, test_1)

        try:
            result.to_csv(self.out_path.replace('/', '\\') + "/predicted_values.csv")
        except:
            print('File write Error')
            self.app_btn_enable()
            return

        self.app_btn_enable()
        self.loadFile_out("/predicted_values.csv")

    def start_predicted_value(self):
        predicted_value_thread = threading.Thread(target=self.predicted_value)
        predicted_value_thread.start()

    def predicted_value(self):
        try:
            d = {}
            ct1 = str(self.comboBox_ct1.currentText())
            ct2 = str(self.comboBox_ct2.currentText())
            ct3 = str(self.comboBox_ct3.currentText())
            brand = str(self.comboBox_brand.currentText())
            shipping = int(self.comboBox_ship.currentText())
            cond = int(self.comboBox_cond.currentText())

            self.app_btn_disable()

            if ct1 != '_':
                d['cat1'] = ct1
            else:
                d['cat1'] = 'Other'
            if ct2 != '_':
                d['cat2'] = ct2
            else:
                d['cat2'] = 'Other'
            if ct3 != '_':
                d['cat3'] = ct3
            else:
                d['cat3'] = 'Other'
            if brand != '_':
                d['brand_name'] = brand
            else:
                d['brand_name'] = 'Others'
            if cond != '_':
                d['item_condition_id'] = cond
            else:
                d['item_condition_id'] = 5
            if shipping != '_':
                d['shipping'] = shipping
            else:
                d['shipping'] = 0

            p = Prediction()
            result = p.predicted_one_value(d)
            self.label_p_price.setText(result)

            # df = self.predicted_for_brands(d)
            # self.load_df_to_table(df)
        except:
            print('Error in predicted_value(), predicted_value() not complete')
        finally:
            self.app_btn_enable()


    def predicted_for_brands(self, d):
        df = pd.DataFrame(columns=['cat1', 'cat2', 'cat3', 'brand_name', 'Price'])
        p = Prediction()
        ct1 = d['cat1']
        ct2 = d['cat2']
        ct3 = d['cat3']
        ct2_df = self.cat_df[self.cat_df['cat1'] == ct1]
        ct3_df = ct2_df[ct2_df['cat2'] == ct2]
        brand_df = ct3_df[ct3_df['cat3'] == ct3]
        for i in set(list(brand_df['brand_name'])):
            d['brand_name'] = i
            result = p.predicted_one_value(d)
            to_append = [d['cat1'], d['cat2'], d['cat3'], str(i), str(result)]
            a_series = pd.Series(to_append, index=df.columns)
            df = df.append(a_series, ignore_index=True)
        return df

    def start_predicted_df(self):
        predicted_df_thread = threading.Thread(target=self.predicted_df)
        predicted_df_thread.start()

    def predicted_df(self):
        self.app_btn_disable()
        self.cat_df = pd.read_csv(".\\cat_table\\cat_df.csv")
        p = Prediction()
        df_test = pd.read_csv(self.ts_path)
        test_1 = p.tidy_up_test_data(df_test, self.cat_df)
        result = p.prediction_df(test_1)
        try:
            result.to_csv(self.out_path.replace('/', '\\') + "/predicted_values.csv")
        except:
            print('File write Error')
            self.app_btn_enable()
            return

        self.app_btn_enable()
        self.loadFile_out("/predicted_values.csv")

    def app_btn_disable(self):
        self.pushButton_Run.setEnabled(False)
        self.pushButton_Run_2.setEnabled(False)
        self.pushButton_Run_3.setEnabled(False)
        self.pushButton_tr.setEnabled(False)
        self.pushButton_ts.setEnabled(False)
        self.pushButton_out.setEnabled(False)
        self.comboBox_ct1.setEnabled(False)
        self.comboBox_ct2.setEnabled(False)
        self.comboBox_ct3.setEnabled(False)
        self.comboBox_brand.setEnabled(False)
        self.comboBox_cond.setEnabled(False)
        self.comboBox_ship.setEnabled(False)

    def app_btn_enable(self):
        self.pushButton_Run.setDisabled(False)
        self.pushButton_Run_2.setDisabled(False)
        self.pushButton_Run_3.setDisabled(False)
        self.pushButton_tr.setDisabled(False)
        self.pushButton_ts.setDisabled(False)
        self.pushButton_out.setDisabled(False)
        self.comboBox_ct1.setDisabled(False)
        self.comboBox_ct2.setDisabled(False)
        self.comboBox_ct3.setDisabled(False)
        self.comboBox_brand.setDisabled(False)
        self.comboBox_cond.setDisabled(False)
        self.comboBox_ship.setDisabled(False)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
