from datetime import datetime
import locale
def date_convert(s, lc_time='pt_BR.UTF-8', cleanup='[/ ]', fmt='%B%Y'):
    current_locale = locale.getlocale(locale.LC_TIME)
    locale.setlocale(locale.LC_TIME, lc_time)
    t = s.replace(cleanup, '', regex=True)\
            .apply(lambda x: None if x == '' else datetime.strptime(x, fmt))
    locale.setlocale(locale.LC_TIME, current_locale)
    return t