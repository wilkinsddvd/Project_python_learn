import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
soup = BeautifulSoup('&lt;p&gt;Hello&lt;/p&gt;', 'lxml')
print(soup.p.string)