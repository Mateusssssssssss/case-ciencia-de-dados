dados_dup = dados.duplicated().sum()
dados_dup1 = dados[['FrqAnual', 'CusInic']].duplicated()
print(f'Quantidade de dados duplicados: {dados_dup}')
print(dados_dup1)