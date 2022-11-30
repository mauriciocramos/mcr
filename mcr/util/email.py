import sys
from os.path import isfile
from glob import glob
from bs4 import BeautifulSoup
from dateutil.parser import parse
from time import time
import re
import pandas as pd
import mailbox
import smtplib
import ssl
from email.header import decode_header

from mcr.util import get_columns_by_content_pattern

ENCODING_PATTERN = re.compile(r'.*(=\?).*(\?=)')
EMAIL_REGEX = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'  # based on http://emailregex.com/
EXACT_MATCH_EMAIL_REGEX = r'(^' + EMAIL_REGEX + r'$)'
PARTIAL_MATCH_EMAIL_REGEX = r'<?(' + EMAIL_REGEX + r')>?'
NEGATIVE_LOOKAHEAD_AMPERSIGN = r'(?!@)' # negative lookahead assertion of the "@"
OPTIONALLY_BRACKET_ENCLOSED_EMAIL_REGEX = r'<?(' + EMAIL_REGEX + r')>?.*?' + NEGATIVE_LOOKAHEAD_AMPERSIGN + '$'
INTERNAL_DOMAINS = re.compile(r'@hostname\.com')
EMAIL_FIELDS = \
    ['X-GM-THRID', 'X-Gmail-Labels', 'Date', 'From', 'To', 'Cc', 'Bcc', 'Subject', 'Content-Type', 'Message-ID']


def sendmail(smtp, port, sender, password, name, recipients, subject, body):
    # reference: https://realpython.com/python-send-email/#starting-a-secure-smtp-connection
    # name = 'Name <name@hostname.com>'
    message = 'Subject: {subject}\nTo: {recipient}\nFrom: {name}\n\n{body}1'
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp, port, context=context) as server:
        try:
            server.login(sender, password)
            for recipient in recipients:
                server.sendmail(sender,
                                recipient,
                                message.format(subject=subject, recipient=recipient, name=name, body=body))
        except smtplib.SMTPAuthenticationError:
            print('SMTPAuthenticationError')


def get_email_columns(df, dtypes='object', pattern=PARTIAL_MATCH_EMAIL_REGEX):
    return get_columns_by_content_pattern(df, dtypes=dtypes, pattern=pattern)


def get_all_fields(input_path):
    """
    Usage: get_all_fields('*.mbox')
    :param input_path: mbox file glob expression
    :return: dataframe containing full file path and field name
    """
    fields = pd.DataFrame({'file_path': [], 'field_name': []})
    for file_path in glob(input_path):
        mbox = mailbox.mbox(file_path)
        message_keys = list(set([key for keys in [message.keys() for message in mbox] for key in keys]))
        mbox.close()
        fields = pd.concat([fields,
                            pd.DataFrame({'file_path': [file_path] * len(message_keys),
                                          'field_name': message_keys})], ignore_index=True)
    return fields


def read_mailboxes(file):
    """
    Reads mailboxes csv and returns a dataframe
    :param file: mailboxes csv file containing fields `file_path`, `mailbox` alias and `email_regex` to match aliases
    :return: mailboxes dataframe indexed by `file_path`
    """
    # reads a csv file containing all mbox file segments per mailbox as exported by Takeout.
    return pd.read_csv(file).set_index('file_path')


def read_emails(files):
    # reads a single or multiple mailboxes in csv format
    emails = \
        pd.concat((pd.read_csv(file, dtype='str', parse_dates=['Date']) for file in glob(files)), ignore_index=True)\
        .set_index('id', verify_integrity=True)
    return emails


def mbox_nobody2dataframe(file_path, fields=None):
    # Naive function that doesn't import email body nor decode any data
    if fields is None:
        fields = EMAIL_FIELDS
    mbox = mailbox.mbox(file_path)
    df = pd.DataFrame([[message[field] for field in fields] for message in mbox], columns=fields)
    mbox.close()
    return df


def header_decoder(header):
    # Decodes email headers such as Subject, From, To, Cc, Bcc
    # Reference: https://github.com/alejandro-g-m/Gmail-MBOX-email-parser/blob/master/parse_mbox.py

    # The header can have several parts encoded in different formats flagged with strings like '=?UTF-8?'
    decoded_header = header
    if header is not None:  # and isinstance(header, Header)
        # try:
        has_encoding = ENCODING_PATTERN.search(str(header)) is not None  # encoding header found
        # except Exception:
        #     print(f'Field type: {type(header)}\nContent:\n{header}')
        #     raise
        if has_encoding:
            decoded_header = ''
            for header_part in decode_header(header):
                # Decode each part ([0]) based on its encoding ([1]) returned by the "decode_header" function
                try:
                    decoded_header += header_part[0].decode('UTF-8' if header_part[1] is None else header_part[1])
                except:
                    if header_part[1] == 'gb2312':  # Error when Codec 'gp2312' simplified chinese
                        decoded_header += header_part[0].decode('gbk')  # fallback to Unified Chinese 'gbk'
                    else:
                        print(f'Field type: {type(header_part[1])}\nContent:\n{header_part[0]}')
                        raise
    # if not type(decoded_header).__name__ in ['str', 'NoneType']:
    #     print(type(decoded_header).__name__)
    #     raise Exception
    return decoded_header


def get_body_parts(message):
    for part in message.get_payload(decode=False):
        if isinstance(part, (list, tuple)):  # list of messages
            for sub_part in get_body_parts(part):
                yield sub_part
        elif part.is_multipart():  # multipart message object
            for sub_part in get_body_parts(part):
                yield sub_part
        else:  # non multipart message
            yield part


def body_part_decoder(part):
    text = None
    if any([x in part.get_content_type() for x in ['text/plain', 'text/html']]):
        text = part.get_payload(decode=True)  # Decode to bytes according to the Content-Transfer-Encoding
        if part.get_content_charset() is None:
            try:
                text = text.decode('utf-8')
            except:
                # TODO: raise a warning
                text = text.decode('iso-8859-1')  # bad bytes for utf-8 naively encoded to iso-8859-1
        else:
            try:
                text = text.decode(part.get_content_charset())  # Decode to string according to content charset
            # Fallback Simplified Chinese 'gb2312' to Unified Chinese 'gbk' otherwise naive 'iso-8859-1'
            except:
                # TODO: raise a warning
                text = text.decode('gbk' if part.get_content_charset() == 'gb2312' else 'iso-8859-1')
        if 'text/html' in part.get_content_type():
            text = BeautifulSoup(text, 'html.parser').get_text(' ', strip=True)
        text = re.sub(r'[ ]{2,}', ' ', text).strip()  # (or r'[\s]{2,}')
        if text == '':
            text = None
    return (text)


def body_decoder(message, exit_on_first_text=True):
    # ref: https://gist.github.com/benwattsjones/060ad83efd2b3afc8b229d41f9b246c4#file-gmail_mbox_parser-py
    decoded_body = []
    if message.is_multipart():
        for part in list(get_body_parts(message)):
            text = body_part_decoder(part)
            if text is not None and (text != ''):
                if exit_on_first_text:
                    return text
                decoded_body.append(text)
    else:
        text = body_part_decoder(message)
        if text is not None and (text != ''):
            if exit_on_first_text:
                return text
            decoded_body.append(text)
    if len(decoded_body) > 0:
        return '\n'.join(decoded_body)


def mbox2dataframe(file_path, fields=None):
    if fields is None:
        fields = EMAIL_FIELDS
    mbox = mailbox.mbox(file_path)
    messages = []
    for msg in mbox:
        message = []
        for field in fields:
            if field in ['From', 'To', 'Cc', 'Bcc', 'X-Gmail-Labels']:
                message.append(header_decoder(msg[field]))
            elif field == 'Subject':
                message.append(str(header_decoder(msg[field])))
            else:
                message.append(msg[field])
        message.append(body_decoder(msg))
        messages.append(message)
    mbox.close()
    df = pd.DataFrame(messages, columns=fields + ['body'])
    return df


def export_all_messages(mailboxes, email_folder, consolidated_file):
    mailboxes = read_mailboxes(mailboxes)
    email_files = [f'{email_folder}{mb}.csv' for mb in mailboxes['mailbox'].unique()]
    if all([isfile(file) for file in email_files]):
        print(f"Files already exported.  To regenerate, first delete the ones necessary: {', '.join(email_files)}")
    else:
        fields = EMAIL_FIELDS
        for group, grouped_df in mailboxes.groupby('mailbox'):  # loop over each mailbox group of files
            emails = pd.DataFrame({})
            skipped = False
            output_file = f'{email_folder}{group}.csv'
            print(f"{'Skipping' if isfile(output_file) else 'Creating'} {output_file}", flush=True)
            for file_path in grouped_df.index:                  # loop over each mailbox file segments
                if isfile(output_file):
                    print(f'\tSkipping {file_path}', flush=True)
                    skipped = True
                else:
                    print(f'\tProcessing {file_path} ...', flush=True)
                    t = time()
                    temp_df = mbox2dataframe(file_path, fields)
                    if emails.shape[0] == 0:
                        emails = temp_df
                    else:
                        emails = pd.concat([emails, temp_df], ignore_index=True)
                    del temp_df
                    print(f'\t\tmailbox to dataframe: {time() - t}', flush=True)
            # No rows mean either an empty mailbox segment file or the whole mailbox were skipped
            if emails.shape[0] == 0:
                if not skipped:
                    print(f'\tNothing to save.  Empty mailbox?', flush=True)
            else:
                # dropping potential duplicates on all columns but the body
                emails = emails.drop_duplicates(subset=fields)

                # region Fix data

                # Fix very rare case where timezone is Eastern Daylight Time and raise an Exception
                # e.g.: "Thu, 2 May 2019 13:19:46 -0400 (Eastern Daylight Time)"
                t = time()
                not_null = emails['Date'].notnull()
                emails.loc[not_null, 'Date'] = emails.loc[not_null, 'Date'].str.replace('Eastern Daylight Time', 'EDT')
                print(f'\tFixing EDT: {time()-t}', flush=True)

                # UnknownTimezoneWarning: tzname CST identified but not understood
                t = time()
                emails.loc[not_null, 'Date'] = emails.loc[not_null, 'Date'].apply(parse, tzinfos={"CST": -6 * 3600})
                print(f'\tdateutil.parser.parse + CST: {time()-t}', flush=True)

                # Feature generation
                emails['mailbox'] = group
                sent = emails['X-Gmail-Labels'].str.contains('Sent|Drafts', case=False, na=False)
                if sent.sum() > 0:
                    emails.loc[sent, 'direction'] = 'outbound'
                if (~sent).sum() > 0:
                    emails.loc[~sent, 'direction'] = 'inbound'

                # endregion

                # saving csv file
                emails.index.name = 'id'
                emails.to_csv(output_file, index=True)
                print(f'\tSaved', flush=True)
        t = time()
        emails = pd.concat((read_emails(file) for file in email_files), ignore_index=True)
        emails.index.name = 'id'
        # ValueError: Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True
        emails['Date'] = pd.to_datetime(emails['Date'], utc=True)
        emails.to_csv(consolidated_file, index=True)
        print(f'pd.to_datetime UTC {consolidated_file}: {time()-t}', flush=True)


if __name__ == '__main__':
    # /data/emails/input/mailboxes.csv
    # /data/emails/output/mailboxes/
    # /data/emails/output/emails.csv
    export_all_messages(sys.argv[1], sys.argv[2], sys.argv[3])
