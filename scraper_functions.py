# # Please install the required Python libraries

# !pip3 install --upgrade pandas # library for manipulating structured data
# !pip3 install --upgrade numpy # library for fundamentals of array computing
# !pip3 install --upgrade requests # library for making request for the static websites
# !pip3 install --upgrade soupsieve  # library to support css selector in beautifulsoup
# !pip3 install --upgrade beautifulsoup4 # a parser that balances between efficiency and leniency
# !pip3 install --upgrade --user lxml # a more efficient parser
# !pip3 install --upgrade html5lib # a parser that acts like a browser, most lenient
# !pip install --upgrade webdriver-manager # library that helps user manage the installation and usage of web drivers # pip3 not supported
# !pip3 install selenium-wire==3.0.6 # library for automating web browser interaction, extended to inspect requests

# Basic libraries
import os
import re
import json
import time
import random
import inspect
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 200

# Web scraping related libraries
import requests
import bs4
from selenium import webdriver # seleniumwire # $$$
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# Other libraries
from collections import OrderedDict
from collections.abc import Iterable
from IPython.display import clear_output

##################################################################################################

def initialize_driver(mode = 'fast', implicit_wait = 10):
    """Initialize and return the web driver.

    Parameters
    ----------
    mode: string (optional, default = 'fast')
        Loading strategy for the web driver, "fast" or "complete".
    implicit_wait: int (optional, default = 10) # seconds
        How many seconds to wait before the driver throw a "No Such Element Exception".

    Returns
    ----------
    seleniumwire.webdriver.browser.Chrome

    """

    caps = DesiredCapabilities().CHROME
    caps['pageLoadStrategy'] = 'eager' if mode == 'fast' else 'normal' if mode == 'complete' else 'none'
    driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities = caps)
    # driver = webdriver.Chrome(executable_path = '/Users/timsmac/chromedriver', desired_capabilities = caps) # $$$
    driver.implicitly_wait(implicit_wait)
    return driver

##################################################################################################

user_name = input('Hi, what is your name?  ')

print('\nNice to meet you, '+user_name+'! Thanks for teaching me scrape...\n')
time.sleep(1)

driver_type = input('Do you need to scrape dynamically loaded website?\n\n(Note choosing "Yes" here will require web-driver installation and scraping progress will be slower. You can also choose to initialize dynamic website scraper later.)\n\nYour choice: [Yes / No]\n\n')

if driver_type.lower()[0] == 'y':

    clear_output()

    print('\n\nPlease choose the loading strategy for your dynamic scraper:\n')
    time.sleep(1)
    dynamic_driver_mode = input('Fast mode: proceed once the basic structure of the web page is loaded.\n\nComplete mode: proceed only after all resources on the page are fully loaded.\n\nYour choice: [Fast / Complete]\n\n')

    print('\n\nOkay, initializing ', end='')
    for i in range(4):
        time.sleep(0.4)
        print('.',end='')

    driver = initialize_driver(mode = dynamic_driver_mode.lower())

    clear_output()

    print('\nDone, the environment is ready. Now Teach Me Scrape!\n')


else:

    print('\n\nOkay, initializing ', end='')
    for i in range(4):
        time.sleep(0.4)
        print('.',end='')

    driver = None

    clear_output()

    print('\nDone, the environment is ready. Now Teach Me Scrape!\n')

##################################################################################################

def ordered_remove_duplicates(li):
    """Remove duplicates and arrange the unique elements by the order of their first occurences.

    Parameters
    ----------
    li: list

    Returns
    ----------
    list

    """
    return list(OrderedDict.fromkeys(li))

def remove_blank_element_in_list(li):
    """Return a cleaned version of the list with all blank elements removed.

    Parameters
    ----------
    li: list

    Returns
    ----------
    list

    """
    return [element for element in li if element.strip()!='']

def flatten_list(l):
    """Flatten a list of lists to a one-layer list (elements are in original order). Note this is NOT recursive, meaning multi-layered list of lists cannot be converted into a single-layered list in one transformation.

    Parameters
    ----------
    l: list

    Returns
    ----------
    list

    """

    return [item for sublist in l for item in sublist]

def robust_flatten_list(li):

    if not any([isinstance(x,(list,tuple)) for x in li]):
        return li # no need to be flatten, no element of the list is list-or-tuple type

    new_li = []
    for x in li:
        if not isinstance(x,(list,tuple)): # if this element is not a list-or-tuple type, make it temporarily wrapped in a singleton list
            new_li.append( [x] )
        else:
            new_li.append( x )

    return flatten_list(new_li)

def deep_flatten_list(li):
    if not any([isinstance(x,(list,tuple)) for x in li]):
        return li # no need to be flatten, no element of the list is list-or-tuple type
    return deep_flatten_list( robust_flatten_list(li) )

def is_iterable(obj):
    """Check if the passed object is iterable.

    Parameters
    ----------
    obj: object

    Returns
    ----------
    boolean

    """

    return isinstance(obj, Iterable)

##################################################################################################
# These functions help us understand the variables that exist in the environment
# which is useful for creating natural language interface for data analysis

def get_local_variables(ignore_underscore = True):
    """Get the name and definition of the local variables.

    Parameters
    ----------
    ignore_underscore : boolean (optional, default = True)
        Whether or not the variables starting with "_" need to be filtered out.

    Returns
    ----------
    dictionary
        A mapping between name and definition of the local variables.

    """
    callers_local_vars = dict(inspect.currentframe().f_back.f_locals.items())
    if ignore_underscore:
        var_keys = list(callers_local_vars.keys())
        for key in var_keys:
            if key.startswith('_'):
                del callers_local_vars[key]
    return callers_local_vars

def get_global_variables(ignore_underscore = True):
    """Get the name and definition of the global variables.

    Parameters
    ----------
    ignore_underscore : boolean (optional, default = True)
        Whether or not the variables starting with "_" need to be filtered out.

    Returns
    ----------
    dictionary
        A mapping between name and definition of the global variables.

    """
    callers_global_vars = dict(inspect.currentframe().f_back.f_globals.items())
    if ignore_underscore:
        var_keys = list(callers_global_vars.keys())
        for key in var_keys:
            if key.startswith('_'):
                del callers_global_vars[key]
    return callers_global_vars

def retrieve_name(var):
    """Retrieve the name of the variable. # Reference https://stackoverflow.com/a/40536047.

    Parameters
    ----------
    var: object
        Variable to get the name of.

    Returns
    ----------
    string
        Name of the variable passed.

    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def get_attributes(obj, ignore_underscore = True):
    """Get a list of valid attributes of the object.

    Parameters
    ----------
    ignore_underscore : boolean (optional, default = True)
        Whether or not the variables starting with "_" need to be filtered out.

    Returns
    ----------
    list
        A list of valid attributes of the object.

    """
    return [x for x in dir(obj) if not x.startswith('_')]

def print_attributes_and_values(obj, ignore_underscore = True):
    """Print the valid attributes of the object and their corresponding values.

    Parameters
    ----------
    ignore_underscore : boolean (optional, default = True)
        Whether or not the variables starting with "_" need to be filtered out.

    Returns
    ----------
    None

    """
    obj_name = retrieve_name(obj)
    attributes = get_attributes(obj, ignore_underscore = ignore_underscore)
    for attr in attributes:
        obj_attr_string = obj_name+'.'+attr
        print(obj_attr_string)
        print(' '*4 + str(eval(obj_attr_string))[:60])
        print('-'*70)

##################################################################################################

def is_readable_content(content):
    """Return whether the content passed is a readable content like Tag or NavigableString; not CData, Comment, Declaration, Doctype, ProcessingInstruction, ResultSet, Script, Stylesheet, XMLFormatter.

    Parameters
    ----------
    content: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    boolean

    """
    # Types that are instances of NavigableString:  CData, Comment, Declaration, Doctype, PreformattedString, ProcessingInstruction, ResultSet, Script, Stylesheet, TemplateString, XMLFormatter
    # Types in the group above that are not String:  CData, Comment, Declaration, Doctype, ProcessingInstruction, ResultSet, Script, Stylesheet, XMLFormatter
    return isinstance(content, (bs4.element.Tag, bs4.element.NavigableString)) and not isinstance(content, (bs4.element.CData, bs4.element.Comment, bs4.element.Declaration, bs4.element.Doctype, bs4.element.ProcessingInstruction, bs4.element.ResultSet, bs4.element.Script, bs4.element.Stylesheet, bs4.element.XMLFormatter))

def get_contents(element):

    """Return a list of non-empty and readable contents/children of the element.

    Parameters
    ----------
    content: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    list of bs4.element

    """
    return [content for content in element.contents if str(content).strip()!='' and is_readable_content(content)]

def get_contents_names(element):
    """Return the list of names of the non-empty and readable contents/children of the element.

    Parameters
    ----------
    content: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    list of string

    """
    return [content.name for content in get_contents(element)]

##################################################################################################

def get_response(url, verbose = True, driver = driver ):
    """Get the response of the HTTP GET request for the target url.

    Parameters
    ----------
    url: string
        The url to the website that needs to be scraped.
    verbose: boolean (optional, default = True)
        Whether or not [Success] message should be printed.
    driver: seleniumwire.webdriver.browser.Chrome (optional, default use global variable driver)
        The web browser driver with which to get the page source of dynamic website.

    Returns
    ----------
    response object or string

    """

    # Static mode when driver is None
    if driver is None:

        headers_list = [{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Referer': 'https://www.google.com/', 'DNT': '1', 'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1'}, {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Referer': 'https://www.google.com/', 'DNT': '1', 'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1'}, {'Connection': 'keep-alive', 'DNT': '1', 'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'document', 'Referer': 'https://www.google.com/'}, {'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9', 'Sec-Fetch-Site': 'same-origin', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-User': '?1', 'Sec-Fetch-Dest': 'document', 'Referer': 'https://www.google.com/'}] # Reference: https://www.scrapehero.com/how-to-fake-and-rotate-user-agents-using-python-3

        try:
            headers = random.choice(headers_list)
            response = requests.get(url, headers = headers)
            response.raise_for_status() # Raise Exception when response was not successful
        except requests.exceptions.HTTPError as http_err:
            print('[Error] HTTP error occurred: '+str(http_err))
            return requests.models.Response() # Return empty response
        except Exception as err:
            print('[Error] Other error occurred: '+str(err))
            return requests.models.Response() # Return empty response
        else:
            if verbose:
                print('[Success] The website at "'+url+'" is collected successfully.')
            return response

    # Dynamic mode when driver is provided
    else:

        if not is_driver_at_url(driver, url):
            go_to_page(driver, url)
            scroll_to_bottom(driver)

        return get_page_source(driver)


def get_soup(response, default_parser = 'lxml'):
    """Get the beautiful soup object of the response object or filepath or html string.

    Parameters
    ----------
    response: requests.models.Response, string
        The response object or filepath or html string.
    default_parser: string (optional, default = lxml)
        Which parser to use when parsing the response.

    Returns
    ----------
    list of response object

    """
    if isinstance(response, requests.models.Response):
        soup = bs4.BeautifulSoup(response.content, default_parser)
    else:
        try:
            soup = bs4.BeautifulSoup(response, default_parser)
        except Exception as err:
            print('[Error] The response object you provided cannot be turned into beautiful soup object: '+str(err))
    return soup

def save_html(html_object , path = '', url = ''): # $$$
    """Save the response or soup object as a HTML file at the path provided.

    Parameters
    ----------
    html_object: requests.models.Response, bs4.BeautifulSoup
        The response or soup object.
    path: string (optional, default = ./TEMP.html)
        The path at which the HTML file will be saved.

    Returns
    ----------
    None

    """
    if path == '':
        if url != '':
            path = './'+re.sub('^https?://','',url).replace('/','_').replace('.','-')+'.html'
        else:
            path = './TEMP.html'
    if isinstance(html_object, requests.models.Response):
        html_text = html_object.text
    elif isinstance(html_object, (bs4.BeautifulSoup,bs4.element.Tag)):
        html_text = str(html_object.prettify())
    else:
        html_text = str(html_object)
    try:
        with open(path,'w') as f:
            f.write(html_text)
            print('[Success] The HTML file is saved succesfully.')
    except Exception as err:
        print('[Error] The response object you provided cannot be turned into beautiful soup object: '+str(err))

def get_response_and_save_html(url, driver = driver, path = ''):
    """Get the response of the website and save it as an HTML.

    Parameters
    ----------
    url: string
        The url to the website that needs to be scraped.
    driver: seleniumwire.webdriver.browser.Chrome (optional, default use global variable driver)
        The web browser driver with which to get the page source of dynamic website.
    path: string (optional, default = ./TEMP.html)
        The path at which the HTML file will be saved.

    Returns
    ----------
    None

    """

    response = get_response(url, driver = driver)

    save_html(response.text, url, path = path)

##################################################################################################

def get_self_index(element):
    """Return the index of the element among its siblings of the same type.

    Parameters
    ----------
    element: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    int

    """
    self_type = element.name
    previous_siblings_of_all_types = list(element.previous_siblings)
    previous_siblings_of_same_type = [element for element in previous_siblings_of_all_types if element.name == self_type]
    return len(previous_siblings_of_same_type) + 1 # css selector starts indexing with 1 instead of 0

def describe_part_of_css_selector(element):
    """Construct part of the css selector path.
    # Reference: https://stackoverflow.com/a/32263260 (basic structure inspiration)
    # Reference: https://csswizardry.com/2012/05/keep-your-css-selectors-short (tips to improve efficiency)

    Parameters
    ----------
    element: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    string

    """

    enough_to_be_unique = False

    element_type = element.name
    element_attrs = element.attrs
    element_attrs_string = ''
    for k,v in element_attrs.items():
        if k == 'id' and str(element_attrs[k])!='':
            element_attrs_string += '#' + element_attrs[k]
            enough_to_be_unique = True
            break
        elif k == 'class' and len(element_attrs[k])>0:
            element_attrs_string += '.'+'.'.join(element_attrs[k])

    element_part = element_type + element_attrs_string

    if not enough_to_be_unique:
        length = get_self_index(element)
        if (length) > 1:
            element_part = '%s:nth-of-type(%s)' % (element_part, length)

    return element_part

def get_css_selector_path(node):
    """Construct the whole css selector path to a certain element.

    Parameters
    ----------
    node: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    string

    """

    path = [describe_part_of_css_selector(node)]
    for parent in node.parents:
        if parent.name == 'html' :
            break
        path.insert(0, describe_part_of_css_selector(parent))
    return ' > '.join(path)

def elevate_css_selector_path(path):
    """Get the css selector path to the element that is one level above the current element.

    Parameters
    ----------
    path: string
        The css selector path to an BS4 element from the parsed tree.

    Returns
    ----------
    string

    """

    return '>'.join(path.split('>')[:-1]).strip() if '>' in path else path


def go_up_multiple_level(orig_path, go_up):
    """Get the css selector path to the element multiple levels up.

    Parameters
    ----------
    orig_path: string
        The css selector path to the source element.
    go_up: int
        The number of levels to go up.

    Returns
    ----------
    string

    """
    path = orig_path[:]
    for i in range(go_up):
        path = elevate_css_selector_path(path)
    return path

def go_up_till_is_tag(element):
    """Return the nearest Tag element, if not itself, return its parent if it is a Tag element.

    Parameters
    ----------
    element: bs4.element
        An BS4 element from the parsed tree.

    Returns
    ----------
    bs4.element.Tag

    """
    if isinstance(element, bs4.element.NavigableString):
        return element.parent
    if isinstance(element, bs4.element.Tag):
        return element
    else:
        print('[Error] Element is still not Tag after getting the parent.')
        return None

##################################################################################################

def get_directly_related_link(element):
    """Extract the link directly related to the element.

    Parameters
    ----------
    element: bs4.element

    Returns
    ----------
    string

    """

    count = 0
    while element.name != 'a' and count < 5:
        element = element.parent
        if element is None:
            return ''
        count += 1
    if element.name != 'a':
        return ''
    else:
        return element.get('href',default='')

def get_indirectly_related_links(element):
    """Extract the links indirectly related to the element (i.e. belonging to the sibling elements).

    Parameters
    ----------
    element: bs4.element

    Returns
    ----------
    list of string

    """

    return remove_blank_element_in_list([link.get('href',default='') for link in element.parent.find_all('a')])

def get_related_link(element):
    """Extract the link directly related to the element, if none is found, get indirectly related links.

    Parameters
    ----------
    element: bs4.element

    Returns
    ----------
    string or list of string

    """

    link = get_directly_related_link(element)

    if link != '':
        return link
    else:
        links = get_indirectly_related_links(element)
        if len(links) == 1 and links[0].strip() != '':
            return links[0]
        else:
            return links

##################################################################################################

def extract_text(element):
    """Extract the textual content of an element.

    Parameters
    ----------
    element: bs4.element

    Returns
    ----------
    string

    """

    return element.getText(separator=u'\n').strip()

def get_longest_separator(text):
    """Return the longest separator (formed by multiple newline) in the text.

    Parameters
    ----------
    text: string

    Returns
    ----------
    string

    """
    if isinstance(text, str) and '\n' in text:
        return max(re.findall(r'\n+', text, re.DOTALL), key=lambda x: len(x))
    else:
        return ''

def recursive_split(text):
    """Return a multi-layer list of lists resulting from a recursive split of the text (split by longer separator first).

    Parameters
    ----------
    text: String
        A piece of text that contains separators of different lengths.

    Returns
    ----------
    list (of lists)

    """
    longest_separator = get_longest_separator(text)
    if longest_separator == '':
        return text
    else:
        return [recursive_split(part) for part in remove_blank_element_in_list(text.split(longest_separator))]

##################################################################################################

def extract_contents(soup, path, verbose = True):
    """Extract and return the texts and links with the target path in the parsed tree.

    Parameters
    ----------
    soup: bs4.soup
        The parsed tree of the response.
    path: string
        The css selector path to the target elements.
    verbose: boolean (optional, default = True)
        Whether or not to print the process message.

    Returns
    ----------
    pd.DataFrame

    """

    if soup is None:
        return None

    if isinstance(soup, pd.DataFrame):
        return soup

    if verbose:
        print('\nExtracting contents ...\n')

    if path.startswith('HEADER:'):
        try:
            tables = pd.read_html(str(soup))
            target_table = [table for table in tables if str(tuple(table.columns.tolist())) == path.replace('HEADER:','')][0]
            return target_table
        except:
            return []

    target_elements = soup.select(path)

    data = pd.DataFrame([(recursive_split(extract_text(target_element)), get_related_link(target_element)) for target_element in target_elements], columns = ['text','url'])

    return data

##################################################################################################

def get_unique_sample_element(soup, target_phrase = '', context_radius = 40):
    """Find and return an element based on the html structure and a target phrase, solicit additional information from user through input questions if needed.

    Parameters
    ----------
    soup: bs4.soup
        The parsed tree of the response.
    target_phrase: string (optional, if not provided, the function will ask user to input)
        The phrase used to find the sample element.
    context_radius: int (optional, default = 40)
        How many characters to display to help user choose recurring phrases based on their contexts.

    Returns
    ----------
    bs4.element.Tag

    """

    target_phrase = target_phrase.lower()
    matched_elements = soup.find_all(text = re.compile(target_phrase,re.IGNORECASE))
    attempt_count = 1

    while len(matched_elements)!=1:

        ################################################################
        # Situation where matched elements have the same textual content

        if len(set([str(matched_element) for matched_element in matched_elements]))==1:
            last_index = -1
            phrases_in_context = []
            whole_page_text = re.sub('\s+',' ',soup.text).lower()

            if whole_page_text.count(target_phrase) == len(matched_elements):

                for i in range(whole_page_text.count(target_phrase)):
                    current_index = whole_page_text.index(target_phrase,last_index+1)
                    phrases_in_context.append(whole_page_text[current_index-context_radius:current_index]+'\\\\ '+whole_page_text[current_index:current_index+len(target_phrase)]+' //'+whole_page_text[current_index+len(target_phrase):current_index+len(target_phrase)+context_radius])
                    last_index = current_index

                if len(set(phrases_in_context))==1:
                    print('[Error] There are '+str(len(phrases_in_context))+' occurences of the same target phrase on the page that have very similar contexts.\nPlease use the browser inspector tool to copy the "selector" or "Selector Path".\n')
                    return None
                else:
                    numbered_contexts = ''
                    for i in range(len(phrases_in_context)):
                        numbered_contexts += 'Choice '+str(i+1)+':  '+phrases_in_context[i] + '\n'
                    print('There are '+str(len(phrases_in_context))+' occurences of the same target phrase on the page,\nplease choose one based on their contexts:\n\n' + numbered_contexts + '\n')

                which_one = 0
                while which_one-1 not in range(len(phrases_in_context)):
                    which_one = input('Which choice is the element you that want to scrape: [1, 2, 3, ...]\n')
                    try:
                        which_one = int(which_one)
                    except:
                        which_one = 0
                matched_elements = [matched_elements[which_one-1]]

            else:
                print('[Error] The number of matched elements and the number of target phrase occurences are not the same.\nPlease use the browser inspector tool to copy the "selector" or "Selector Path".\n')
                return None

        ###########################################################
        if len(matched_elements) > 0 and len(matched_elements) < 5:
            # List numbered choices
            numbered_choices = ''
            for i in range(len(matched_elements)):
                numbered_choices += '\tChoice '+str(i+1)+':  '+str(matched_elements[i])[:80]+ '\n'

            print('\nThere are '+str(len(matched_elements))+' matched elements given your last input. They are:\n'+numbered_choices)

            # Choose one
            which_one = 0
            while which_one-1 not in range(len(matched_elements)):
                which_one = input('Which choice is the element you that want to scrape: [1, 2, 3, ...]\n')
                try:
                    which_one = int(which_one)
                except:
                    which_one = 0
            matched_elements = [matched_elements[which_one-1]]

        else:
            if len(matched_elements) > 5:
                print('\nThere are '+str(len(matched_elements))+' matched elements given your last input. They are:\n\n\t'+'\n\t'.join([str(matched_element)[:80] for matched_element in matched_elements[:10]])+'\n\nPlease be more specific in your target phrase.\n')
            if len(matched_elements) == 0:
                print('\nNo match was found, please check for typos in the target phrase (case insensitive) or check if the website is fully collected.')

            # Search again
            target_phrase = input('What is the displayed text for one of the elements you want to scrape: '+('(Type "QUIT" to stop)' if attempt_count>3 else '')+'\n')
            if target_phrase == 'QUIT':
                print('\n[Error] It is likely that the website is not fully collected.\n        Please try this command: get_response_and_save_html(PUT_IN_YOUR_URL)\n        A HTML file will be created in your local folder, open it with a browser.\n        If you cannot see what you want to find on the page, please switch to dynamic scraping method.\n')
                return None
            matched_elements = soup.find_all(text = re.compile(target_phrase,re.IGNORECASE))

        # Increment attempt count
        attempt_count += 1

    # Match is found by this point
    sample_element = matched_elements[0]
    sample_element = go_up_till_is_tag(sample_element)
    print('\nUnique match is found:\n'+str(sample_element)[:100]+ (' ......' if len(str(sample_element))>100 else '') +'\n\n')

    # If the sample element is script tag, handle it differently
    if sample_element.name == 'script':
        matched_lines = [line for line in sample_element.prettify().split('\n') if target_phrase in line.lower()]
        try:
            assert(len(matched_lines)==1)
            matched_line = matched_lines[0].strip().strip(';')
            matched_data = matched_line.split('=',maxsplit=1)[1].strip()
            data = pd.DataFrame(json.loads(matched_data))
            return data
        except:
            print('[Error] There are multiple occurences of the target phrase in the JS script.\nPlease use another more unique target phrase or inspect the page source for the data in JS script.\n')
            return None

    return sample_element

##################################################################################################

def scrape_what_from_where(target_phrase, url, driver = driver, go_up = 0):
    """Get the contents that are similar to the element with phrase "what" in the website "where".

    Parameters
    ----------
    target_phrase: string
        The displayed text of one of the elements you want to scrape.
    url: string
        The url of the website you want to scrape.
    go_up: int (optional, default = 0)
        How many levels to go up in order to get the amount of contents you want.
    driver: seleniumwire.webdriver.browser.Chrome (optional, default use global variable driver)
        The web browser driver with which to get the page source of dynamic website.

    Returns
    ----------
    pd.DataFrame

    """
    # Get response
    response = get_response(url, driver = driver)

    # Get parse tree
    soup = get_soup(response)

    # Check if the data is in a table, if so, directly return the table
    try:
        tables = pd.read_html(str(soup))
    except:
        tables = []

    if len(tables)>0 and (len(set([tuple(table.columns.tolist()) for table in tables])) == len(tables)):
        tables_containing_target_phrase = [table for table in tables if target_phrase in str(table)]
        tables_containing_target_phrase = sorted(tables_containing_target_phrase, key=lambda t: len(str(t)))
        if len(tables_containing_target_phrase)>0:
            while len(tables_containing_target_phrase)>0:
                print('\nThere are '+str(len(tables_containing_target_phrase))+' tables with the target phrase:\n')
                target_table = tables_containing_target_phrase[0]
                print(target_table)
                is_right_table = input('\nIs this table what you want to scrape? [Yes/No]\n')
                if is_right_table.lower()[0] == 'y':
                    right_header = tuple(target_table.columns.tolist())
                    print('\nThe right header is:\n\t'+str(right_header))
                    return target_table, 'HEADER:'+str(right_header)
                else:
                    tables_containing_target_phrase.pop(0)
            if len(tables_containing_target_phrase)==0:
                print('\nThe target data is not one of the tables, moving on to other html elements.\n')


    # Pinpoint the sample element through dialogue
    sample_element = get_unique_sample_element(soup, target_phrase)
    if sample_element is None:
        return None, ''
    if isinstance(sample_element, pd.DataFrame):
        print('[Success] Data is in the JS script and now extracted as a DataFrame into the variable "soup".\n')
        return sample_element, ''

    # Build the css selector path to the sample element
    sample_path = get_css_selector_path(sample_element)

    # Go up the parse tree if needed:
    path = go_up_multiple_level(sample_path, go_up = go_up)

    # Extract content
    data = extract_contents(soup, path)

    # If data is extracted from html path instead of from json, print the path for future use
    if path != '':
        print('\n[Success] The selector path used to extract contents is:\n\n\t'+path+'\n')

    return data, path

##################################################################################################

def create_page_url_list(template_url, start_index, end_index, unique_first_url = None):
    """Generate a list of urls to scrape from.

    Parameters
    ----------
    template_url: string
        The url template with placeholder "NUMBER" that will be replaced by index number.
    start_index: int
        The first index to be plugged into the template.
    end_index: int
        The last index to be plugged into the template.
    unique_first_url: string (optional, default = None)
        If the first web page in the pagination process has a different format compared the one after, provide it here.

    Returns
    ----------
    list of string

    """

    page_url_list = []
    if unique_first_url is not None:
        page_url_list.append(unique_first_url)
    for i in range(start_index,end_index+1):
        page_url_list.append(template_url.replace('NUMBER',str(i)))
    return page_url_list

def get_base_url(url):
    """Get the base url from a url path.

    Parameters
    ----------
    url: string
        Any url path of the website.

    Returns
    ----------
    string

    """
    return url.split('://')[0]+'://'+url.split('://')[1].split('/')[0]

def scrape_path_from_pages(path, pages, driver = driver, save_separately = False, file_path_template = None , reporting_interval = None, verbose = False, return_list_of_tables = False): # $$$
    """Get the contents that are located with path on the pages provided. A batch version of the scrape_what_from_where function.

    Parameters
    ----------
    path: string
        The css selector path to the target contents.
    pages: list of string
        The urls to the web pages to be scraped.
    driver: seleniumwire.webdriver.browser.Chrome (optional, default use global variable driver)
        The web browser driver with which to get the page source of dynamic website.
    save_separately: boolean (optional, default = False)
        Whether or not to save the data from each page separately.
    file_path_template: string (optional, default = None)
        If save_separately is True, where to save the data files. It should contain the placeholder "NUMBER" which will be replaced by index number.
    reporting_interval: int (optional, default = None)
        After how many pages should a progress message be printed.
    verbose: boolean (optional, default = False)
        Whether to print detailed scraping progress messages.

    Returns
    ----------
    pd.DataFrame

    """

    number_of_pages = len(pages)
    index_width = len(str(number_of_pages+1))

    if reporting_interval is None:
        reporting_interval = int(number_of_pages/10)+1 if number_of_pages<1000 else int(number_of_pages/40)

    output_dataframe = pd.DataFrame()
    output_dataframe_list = []

    for i in range(number_of_pages):

        time.sleep(0.15) # $$$

        if i % reporting_interval == 0:
            print(str(i)+'/'+str(number_of_pages), end=', ')
        url = pages[i]

        dataframe = extract_contents(get_soup(get_response(url, verbose = verbose, driver = driver)), path, verbose = verbose)

        if save_separately:
            if file_path_template is None:
                print('\n[Error] To save the dataframes from different pages separatorly, you need to provide a file path template.\n')
                return None
            file_path = file_path_template.replace('NUMBER', str(i).zfill(index_width))
            dataframe.to_csv(file_path, index = False)

        else:
            if return_list_of_tables:
                output_dataframe_list.append(dataframe)
            else:
                output_dataframe = output_dataframe.append(dataframe, ignore_index=True)

    print('\n\n[Success] Content extraction finished.\n\n')

    if return_list_of_tables:
        return output_dataframe_list
    if save_separately:
        return None
    return output_dataframe

##################################################################################################

def terminate_driver(driver_instance):
    """Terminate the web driver and delete the driver variable.

    Parameters
    ----------
    driver: seleniumwire.webdriver.browser.Chrome
        An active web driver instance.

    Returns
    ----------
    boolean

    """

    try:
        if driver_instance is not None:
            driver_instance.quit()
        global driver
        del driver
        return True
    except Exception as err:
        print('[Error] Web driver failed to quit properly. '+str(err))
        return False

def go_to_page(driver, url):
    """Initialize and return the web driver.

    Parameters
    ----------
    implicit_wait: int (optional, default = 10) # seconds
        How many seconds to wait before the driver throw a "No Such Element Exception"

    Returns
    ----------
    seleniumwire.webdriver.browser.Chrome

    """

    try:
        driver.get(url)
        return True
    except Exception as err:
        print('[Error] '+str(err))
        return False

def get_page_source(driver):
    """Return the page source of the web page that the driver is currently at.

    Parameters
    ----------
    driver: seleniumwire.webdriver.browser.Chrome
        An active web driver instance.

    Returns
    ----------
    boolean

    """
    return driver.page_source

def is_driver_at_url(driver, url):
    """Return whether the driver is at the url provided.

    Parameters
    ----------
    driver: seleniumwire.webdriver.browser.Chrome
        An active web driver instance.
    url: string
        A url to check if the driver is at.

    Returns
    ----------
    boolean

    """
    stripped_url = re.sub('https?://','', url.strip('/'))
    stripped_current_url = re.sub('https?://','', driver.current_url.strip('/'))
    return stripped_current_url == stripped_url

def scroll_to_bottom(driver, scroll_pause_time = 1):
    """Make the driver scroll to the bottom of the page step by step by a certain pause interval.
    # Reference: https://stackoverflow.com/a/28928684/1316860
    # Reference: https://stackoverflow.com/a/43299513

    Parameters
    ----------
    driver: seleniumwire.webdriver.browser.Chrome
        An active web driver instance.
    scroll_pause_time: int (optional, default = 1)
        Time to wait between scroll actions.

    Returns
    ----------
    boolean

    """

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(scroll_pause_time)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

##################################################################################################

def get_tables_with_links():
  tables = pd.read_html(get_page_source(driver))
  link_groups = [[[a['href'] for a in tr.findAll('a') if a.has_attr('href')] for tr in table_body.findAll('tr')] for table_body in get_soup(get_page_source(driver)).findAll('tbody')]
  assert(len(tables)==len(link_groups))
  tables_with_links = []
  for i in range(len(tables)):
    table = tables[i]
    table['Links'] = link_groups[i]
    tables_with_links.append(table)
  return tables_with_links

