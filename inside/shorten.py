# Commented out statements was from the first pass

exit(0)    # NOTE This prevents accidental execution

# NOTE update these 2 file names before use
with open("large_files/LLS_DDOS_2.0.2-inside-SHORT") as data:
    fi = open("LLS_DDOS_2.0.2-inside-all-MORE", "w") 
    
    prev = ""
    write = False
#    write = True
    for line in data:
        if " 31000 " in line or " 55000 " in line or " 88000 " in line: # start position
            write = True
            fi.write(prev)      # will write previous line from file
        if " 35000 " in line or " 58000 " in line or " 94000 " in line: # stop position
            write = False
        if write:
            fi.write(str(line))
        prev = line

    fi.close()

# OG 'all' file:
# write lines 31950 to 32021
#             56200 to 56451
#             90150 to 90251
