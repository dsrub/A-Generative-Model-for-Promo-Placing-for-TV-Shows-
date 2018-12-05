import psycopg2
import pandas as pd
import re

# Author:
# Douglas Rubin

def redshift_connect(dbname, host, port, user, pswd):

	"""
	connect to redshift database
	
	Inputs: all strings to connect to the redshift database
	Output: a tuple the cursor for the connection and the connection object
	"""

	con=psycopg2.connect(dbname = dbname, host = host, 
	port = port, user = user, password = pswd)

	return (con.cursor(), con)


def redshift_secure_connect(dbname, host, port, user):

	"""
	connect to redshift database securely.  Using PGPASS file set up on the machine.  PGPASS located in ~/.pgpass
    contains info about connections, like host:port:dbname:username:password
    for all DBs you connect to, such that you do not need to include password in cleartext call in a notebook or program
	
	Inputs: all strings to connect to the redshift database
	Output: a tuple the cursor for the connection and the connection object
	"""

	con=psycopg2.connect(dbname = dbname, host = host, 
	port = port, user = user)

	return (con.cursor(), con)


def pd_df(cur, sql_command):

	"""
	define a function that takes in an SQL command (as a string) and returns a pd dataframe
    (be careful not return anything too huge)
    
    Inputs: cur - the cursor object instantiated and returned by the redshift_connect funciton
    		sql_command - a string, the SQL command to query the database
    Output: a pandas dataframe containing the data from your query
    """

    ### execute sql command
	cur.execute(sql_command)
    
    ### get list of columns
	col_list = [x[0] for x in cur.description]
    
    ### fetch all data returned
	data = cur.fetchall()
    
    ### return pd dataframe
    
	return pd.DataFrame(data, columns=col_list)


def make_bins(cur, col_name, n_bins, table):

	"""
	a function that bins data of a certain column from a table in redshift and 
	returns a list of the bin centersand a list of the bin count
	
    Inputs: cur - the cursor object instantiated and returned by the redshift_connect funciton
    		col_name - the column you want to bin (string)
    		n_bins - the number of bins you want
    		table - the table in which the column lives (string)

    Output: A tuple of 2 lists.  The first list gives the centers of the bins and the second
    		gives the count for each bin
	"""
    
    ### retrieve max value of column
	cur.execute('select max('+col_name+') from ' +table+';')
	max_val = float(cur.fetchall()[0][0])
    
    ### retrieve min value of column
	cur.execute('select min('+col_name+') from ' +table+';')
	min_val = float(cur.fetchall()[0][0])
    
	incr = (max_val-min_val)/n_bins
	incr_str = str(incr)

    ### define relevent query
	x = """select
            bucket_floor,
            bucket_ceiling,
            bucket_floor || ' to ' || bucket_ceiling as bucket_name,
            count(*) as cnt
       from (
             select 
                floor(""" + col_name+"""/"""+incr_str+""")*"""+incr_str+""" as bucket_floor,
                floor(""" + col_name+"""/"""+incr_str+""")*"""+incr_str+""" +"""+incr_str+ """ as bucket_ceiling
             from """+table+""") a
       group by 1, 2
       order by 1;"""

	x = re.sub(' +',' ',x.replace('\n',' ')).strip()
    
	hist_df = pd_df(cur, x)
    
	hist_df['bucket_floor']=hist_df['bucket_floor'].apply(float)
	hist_df['bucket_ceiling']=hist_df['bucket_ceiling'].apply(float)
	hist_df['cnt']=hist_df['cnt'].apply(int)
    
	hist_df['bin_centers'] = hist_df['bucket_floor'] + (hist_df['bucket_ceiling'][0]-hist_df['bucket_floor'][0])/2.
    
	bin_center_list = list(hist_df['bin_centers'])
	cnt_list = list(hist_df['cnt'])
    
	return (bin_center_list, cnt_list)

def close_conn(cur, con):

	"""
	close up cursor and close up the connection when you are finished
	"""

	cur.close()
	con.close()
    
    