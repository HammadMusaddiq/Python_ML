import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

import getpass
import os, time



subject = "Email with Python"

msg_content = """Hi,<br><br>

I hope this email finds you well. <br><br>

Best wishes & Regards<br>
Hammad<br>
"""

sender_email = "hammadm@gmail.com"
receiver_email = ["hammad1@gmail.com", "hammad@hotmail.com"] # you can add multiple emails in this list using comma's in bettween

cc_email = []
bcc_email = []
password_email = getpass.getpass("Please enter email password: ")  #Here put pasword to your eventim login email. 

path_to_folder = 'D:\\New Folder\\' # all files from this folder will be sent in email 

def attatchment_files():
    file_names = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]
    return file_names

def Email(r_email):
    # Create a multipart message and set headers

    hotmail_smtp = "smtp.office365.com" #587
    gmail_smtp = "smtp.gmail.com" #465

    message = MIMEMultipart('mixed')
    message['From'] = sender_email
    message['To'] = ", ".join(r_email)
    message['CC'] = ", ".join(cc_email)
    message['Bcc'] = ", ".join(bcc_email)
    message['Subject'] = subject
    message['Content-Type'] = "text/html"

    body = MIMEText(msg_content, 'html')
    message.attach(body)

    file_names = attatchment_files()

    try:
        for path in file_names:  # add files to the message
            file_path = os.path.join(path_to_folder, path)
            file_type = path.split('.')[-1]
            attachment = MIMEApplication(open(file_path, "rb").read(), _subtype = file_type) 
            attachment.add_header('Content-Disposition','attachment', filename = path)
            message.attach(attachment)
            
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(gmail_smtp, 465, context=context) as server:
            server.login(sender_email, password_email)
            server.sendmail(sender_email, r_email + cc_email + bcc_email, message.as_string())
            server.quit()
        
    except Exception as E:
        print(E)


if __name__ == "__main__":

    for mail in receiver_email:
        Email([mail])
        print("sent")
        time.sleep(2)
