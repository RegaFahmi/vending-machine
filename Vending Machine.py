import sys
import cv2
import numpy as np
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from skimage.metrics import structural_similarity

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('Vending Machine.ui', self)

        # Inisialisasi variabel
        self.Image = None
        self.item_counts = {
            "Coca Cola": 0,
            "Sprite": 0,
            "Tebs": 0,
            "Aqua": 0,
            "Nescafe": 0,
            "Lasegar": 0
        }

        self.item_prices = {
            "Coca Cola": 10000,
            "Sprite": 10000,
            "Tebs": 7000,
            "Aqua": 5000,
            "Nescafe": 7000,
            "Lasegar": 7000
        }

        self.total_price = 0
        self.detected_value = 0

        # Menghubungkan tombol dengan metode yang sesuai
        self.cola_Beli.clicked.connect(self.colabeli)
        self.sprite_Beli.clicked.connect(self.spritebeli)
        self.tebs_Beli.clicked.connect(self.tebsbeli)
        self.aqua_Beli.clicked.connect(self.aquabeli)
        self.nescafe_Beli.clicked.connect(self.nescafebeli)
        self.lasegar_Beli.clicked.connect(self.lasegarbeli)
        self.deteksi_btn_2.clicked.connect(self.deteksi_koin)
        self.deteksi_btn.clicked.connect(self.deteksi_kertas)
        self.respesan_btn.clicked.connect(self.reset_pesan)
        self.ressaldo_btn.clicked.connect(self.reset_saldo)
        self.bayar_btn.clicked.connect(self.bayar)

    # Metode untuk mendeteksi koin pada gambar
    def deteksi_koin(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, 'Open File', 'DATASET_KOIN', 'Image files (*.jpg *.png *.jpeg)')
        if imagePath:
            self.Image = cv2.imread(imagePath)
            cv2.imshow('Original', self.Image)

        if self.Image is not None:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)

            img = self.Image

            # Otsu Tresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imshow('Otsu', thresh)

            # Dilasi
            kernel = np.ones((3, 3), np.uint8)
            sure_bg = cv2.dilate(thresh, kernel, iterations=3)
            self.Image = sure_bg
            cv2.imshow('Dilasi', sure_bg)

            # Kontur
            contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            total_value = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                print(area)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))

                if area > 500 and circularity > 0.7:
                    if area < 16100:
                        value = 100
                    elif area < 17020:
                        value = 1000
                    elif area < 20000:
                        value = 200
                    else:
                        value = 500

                    total_value += value
                    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
                    self.Image = cv2.putText(img, str(value), (cnt.ravel()[0], cnt.ravel()[1]),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.detected_value += total_value
            formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
            self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
            cv2.imshow('Hasil', self.Image)

    def deteksi_kertas(self):
        imagePath, _ = QFileDialog.getOpenFileName(self, 'Open File', 'FOTO_UANG', 'Image files (*.jpg *.png *.jpeg)')
        if imagePath:
            self.Image = cv2.imread(imagePath)
            cv2.imshow('Original', self.Image)

        if self.Image is not None:
            dataset_path = 'DATASET_UANG'
            templates = {
                1000: f'{dataset_path}/1RIBU/',
                2000: f'{dataset_path}/2RIBU/',
                5000: f'{dataset_path}/5RIBU/',
                10000: f'{dataset_path}/10RIBU/',
                20000: f'{dataset_path}/20RIBU/',
                50000: f'{dataset_path}/50RIBU/',
                75000: f'{dataset_path}/75RIBU/',
                100000: f'{dataset_path}/100RIBU/'
            }

            gray_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

            best_match = None
            best_val = 0

            for nominal, folder in templates.items():
                for filename in os.listdir(folder):
                    template_path = os.path.join(folder, filename)
                    template = cv2.imread(template_path, 0)
                    if template is None:
                        continue

                    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    if max_val > best_val:
                        best_val = max_val
                        best_match = nominal

            if best_match is not None:
                if best_match == 100000:
                    self.ssim100()
                elif best_match == 50000:
                    self.ssim50()
                elif best_match == 1000:
                    self.ssim1()
                elif best_match == 10000:
                    self.ssim10()
                else:
                    self.detected_value += best_match
                    formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
                    self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
                    print(f'Detected: Rp. {best_match}')
            else:
                self.textBrowser_2.setText("No match found")
                QMessageBox.warning(self, 'Warning', 'UANG PALSU')
                print('No match found')

    def ssim1(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray

        # Load the sample image and convert to grayscale
        sample = cv2.imread('DATASET_UANG/1RIBU/16.jpg')
        if sample is None:
            raise ValueError("Sample image not found or unable to load.")
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        test = self.Image

        # Compute SSIM between the sample and the test images
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM score:", score)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        # Display images
        cv2.imshow('sample', sample_gray)
        cv2.imshow('citrapengujian', test)
        cv2.imshow('perbedaan menggunakan negative filter', diff)
        cv2.imshow("mask", mask)
        cv2.imshow("filled after", filled_after)
        self.Image = filled_after

        cv2.imshow("Sample", sample)
        cv2.imshow("Test", test)
        cv2.imshow("Difference using negative filter", diff)

        cv2.waitKey(0)

        # Menampilkan pop up berdasarkan skor SSIM
        if score < 0.95:
            QMessageBox.warning(self, 'Peringatan', 'Uang yang diinputkan adalah palsu')
        else:
            self.detected_value += 1000
            formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
            self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
            print(f'Detected: Rp. 1000')

    def ssim5(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray

        # Load the sample image and convert to grayscale
        sample = cv2.imread('DATASET_UANG/5RIBU/16.jpg')
        if sample is None:
            raise ValueError("Sample image not found or unable to load.")
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        test = self.Image

        # Compute SSIM between the sample and the test images
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM score:", score)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        # Display images
        cv2.imshow('sample', sample_gray)
        cv2.imshow('citrapengujian', test)
        cv2.imshow('perbedaan menggunakan negative filter', diff)
        cv2.imshow("mask", mask)
        cv2.imshow("filled after", filled_after)
        self.Image = filled_after

        cv2.imshow("Sample", sample)
        cv2.imshow("Test", test)
        cv2.imshow("Difference using negative filter", diff)

        cv2.waitKey(0)

        # Menampilkan pop up berdasarkan skor SSIM
        if score < 0.95:
            QMessageBox.warning(self, 'Peringatan', 'Uang yang diinputkan adalah palsu')
        else:
            self.detected_value += 5000
            formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
            self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
            print(f'Detected: Rp. 5000')
    def ssim100(self):  # fungsi ssim 100%
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray

        # Load the sample image and convert to grayscale
        sample = cv2.imread('DATASET_UANG/100RIBU/16.jpg')
        if sample is None:
            raise ValueError("Sample image not found or unable to load.")
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        test = self.Image

        # Compute SSIM between the sample and the test images
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM score:", score)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        # Display images
        cv2.imshow('sample', sample_gray)
        cv2.imshow('citrapengujian', test)
        cv2.imshow('perbedaan menggunakan negative filter', diff)
        cv2.imshow("mask", mask)
        cv2.imshow("filled after", filled_after)
        self.Image = filled_after

        cv2.imshow("Sample", sample)
        cv2.imshow("Test", test)
        cv2.imshow("Difference using negative filter", diff)

        cv2.waitKey(0)

        # Menampilkan pop up berdasarkan skor SSIM
        if score < 0.97:
            QMessageBox.warning(self, 'Peringatan', 'Uang yang diinputkan adalah palsu')
        else:
            self.detected_value += 100000
            formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
            self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
            print(f'Detected: Rp. 100000')

    def ssim50(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray

        # Load the sample image and convert to grayscale
        sample = cv2.imread('DATASET_UANG/50RIBU/16.jpg')
        if sample is None:
            raise ValueError("Sample image not found or unable to load.")
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        test = self.Image

        # Compute SSIM between the sample and the test images
        (score, diff) = structural_similarity(sample_gray, test, full=True)
        print("SSIM score:", score)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(sample.shape, dtype='uint8')
        filled_after = test.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        # Display images
        cv2.imshow('sample', sample_gray)
        cv2.imshow('citrapengujian', test)
        cv2.imshow('perbedaan menggunakan negative filter', diff)
        cv2.imshow("mask", mask)
        cv2.imshow("filled after", filled_after)
        self.Image = filled_after

        cv2.imshow("Sample", sample)
        cv2.imshow("Test", test)
        cv2.imshow("Difference using negative filter", diff)

        cv2.waitKey(0)

        # Menampilkan pop up berdasarkan skor SSIM
        if score < 0.97:
            QMessageBox.warning(self, 'Peringatan', 'Uang yang diinputkan adalah palsu')
        else:
            self.detected_value += 50000
            formatted_total_value = f"{self.detected_value:,}".replace(',', '.')
            self.textBrowser_2.setText(f"Rp. {formatted_total_value}\n")
            print(f'Detected: Rp. 50000')

    # Metode untuk memperbarui daftar pembelian
    def update_list_widget(self, item_name):
        self.item_counts[item_name] += 1
        item_text = f"{item_name} - {self.item_counts[item_name]}"
        self.total_price += self.item_prices[item_name]

        items = self.listWidget.findItems(item_name, QtCore.Qt.MatchStartsWith)
        if items:
            item = items[0]
            item.setText(item_text)
        else:
            self.listWidget.addItem(item_text)

        formatted_price = f"{self.total_price:,}".replace(',', '.')
        self.textBrowser.setText(f"Rp. {formatted_price}")

    # Metode untuk membeli Coca Cola
    def colabeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Coca Cola")

    # Metode untuk membeli Sprite
    def spritebeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Sprite")

    # Metode untuk membeli Tebs
    def tebsbeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Tebs")

    # Metode untuk membeli Aqua
    def aquabeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Aqua")

    # Metode untuk membeli Nescafe
    def nescafebeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Nescafe")

    # Metode untuk membeli Lasegar
    def lasegarbeli(self):
        if not self.textBrowser_2.toPlainText().strip():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Masukan Uang Terlebih Dahulu")
            msg.setStyleSheet(
                "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
            msg.setText("Masukan uang terlebih dahulu sebelum membeli.")
            msg.exec_()
            return

        self.update_list_widget("Lasegar")

    # Metode untuk mereset pesan pembelian
    def reset_pesan(self):
        self.listWidget.clear()
        self.textBrowser.clear()
        self.total_price = 0
        for item in self.item_counts:
            self.item_counts[item] = 0

    # Metode untuk mereset pesan nominal uang
    def reset_saldo(self):
        self.textBrowser_2.clear()
        self.detected_value = 0

    # Metode untuk proses pembayaran
    def bayar(self):
        saldo_text = self.textBrowser_2.toPlainText().strip().replace("Rp. ", "").replace(".", "")
        total_text = self.textBrowser.toPlainText().strip().replace("Rp. ", "").replace(".", "")

        if saldo_text and total_text:
            saldo = int(saldo_text)
            total = int(total_text)

            confirmation_message = "Apakah Anda yakin ingin membeli?"
            confirmation = QMessageBox.question(self, "Konfirmasi Pembelian", confirmation_message,
                                                QMessageBox.Yes | QMessageBox.No)

            if confirmation == QMessageBox.Yes:
                if saldo >= total:
                    new_saldo = saldo - total
                    formatted_new_saldo = f"Rp. {new_saldo:,}".replace(',', '.')
                    self.textBrowser_2.setText(formatted_new_saldo)
                    self.reset_pesan()
                    self.textBrowser_2.clear()
                    self.detected_value = 0

                    remaining_balance_message = f"Berhasil Membeli\nKembalian: {formatted_new_saldo}"
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Pembayaran Berhasil")
                    msg.setStyleSheet(
                        "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
                    msg.setText(remaining_balance_message)
                    msg.exec_()
                else:
                    insufficient_balance_message = "Uang kamu tidak cukup untuk melakukan pembayaran."
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Uang Tidak Cukup")
                    msg.setStyleSheet(
                        "QMessageBox { background-color: white; color: black; } QPushButton { background-color: blue; color: white; }")
                    msg.setText(insufficient_balance_message)
                    msg.exec_()


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Vending Machine')
window.show()
sys.exit(app.exec_())

