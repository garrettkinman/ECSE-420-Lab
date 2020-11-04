using CSV
using XLSX

data = CSV.read("output.txt")

XLSX.writetable("output.xlsx", data)
