import smtplib, ssl

class Mailer:

    def __init__(self):
        # Enter your email below. This email will be used to send alerts.
        # E.g., "email@gmail.com"
        self.EMAIL = "xxx@gmail.com"
        # Enter the email password below. Note that the password varies if you have secured
        # 2 step verification turned on. You can refer the links below and create an application specific password.
        # Google mail has a guide here: https://myaccount.google.com/lesssecureapps
        # For 2 step verified accounts: https://support.google.com/accounts/answer/185833
        # Example: adoiwahfeoowhq
        self.PASS = ""
        self.PORT = 465
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)

    def send(self, mail, nframes, seconds, pos_certainty, neg_certainty):
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)
        # message to be sent
        SUBJECT = 'Predictions alert!'
        TEXT = f'Total frames detected: {nframes} for approximately in time: {seconds} sec. \n > Class1 probability: {pos_certainty}. \n > Class2 probability: {neg_certainty}. \n > Prob. difference between both classes: {pos_certainty - neg_certainty}.'
        message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

        # sending the mail
        self.server.sendmail(self.EMAIL, mail, message)
        self.server.quit()
