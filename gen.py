with open("password.txt", "w+") as f:
    st, ed = 0, 2000
    for i in range(st, st + ed):
        f.write(str(i) + "\n")
    f.write(str(st + ed)) 
    f.close()