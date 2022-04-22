import pandas as pd

dados = pd.read_csv("/home/brunohelghast/PROFISSIONAL/PYTHON/SCIKIT_LEARN/previsaoAprovacaoCartaoCredito/application_record.csv")
dados = dados.drop(columns=["ID"])
dados.drop_duplicates(inplace=True)
dados = dados.dropna()

bom = dados["CLASSIFICAO"] == "bom"
mal = dados["CLASSIFICAO"] == "mal"

masculino = dados["CODE_GENDER"] == "M"
feminino = dados["CODE_GENDER"] == "F"

temCarro = dados["FLAG_OWN_CAR"] == "Y"
temCasa = dados["FLAG_OWN_REALTY"] == "Y"

masculinoBom = (masculino == bom).value_counts()
masculinoMal = (masculino == mal).value_counts()
masculinoBomTemCasa = (masculino == bom) == (masculino == temCasa)

print(masculinoMal, masculinoBom)
print(masculinoBomTemCasa.value_counts())

rendaWorking = dados["NAME_INCOME_TYPE"] == "Working"
rendaComerAssociate = dados["NAME_INCOME_TYPE"] == "Commercial associate"
rendaStateServant = dados["NAME_INCOME_TYPE"] == "State servant"
rendaPensioner = dados["NAME_INCOME_TYPE"] == "Pensioner"
rendaStudent = dados["NAME_INCOME_TYPE"] == "Student"

ensinoSuperior = dados["NAME_EDUCATION_TYPE"] == "Higher education"
ensinoSuperiorImc = dados["NAME_EDUCATION_TYPE"] == "Incomplete higher"
ensinoMedio = dados["NAME_EDUCATION_TYPE"] == "Secondary / secondary special"
ensinoMedioInferior = dados["NAME_EDUCATION_TYPE"] == "Lower secondary"
ensinoAcademico = dados["NAME_EDUCATION_TYPE"] == "Academic degree"

ocupacaoLaborers = dados["OCCUPATION_TYPE"] == "Laborers"
ocupacaoSales = dados["OCCUPATION_TYPE"] == "Sales staff"
ocupacaoCoreStaff = dados["OCCUPATION_TYPE"] == "Core staff"
ocupacaoManagers = dados["OCCUPATION_TYPE"] == "Managers"
ocupacaoDrivers = dados["OCCUPATION_TYPE"] == "Drivers"
ocupacaoHighSkillTech = dados["OCCUPATION_TYPE"] == "High skill tech staff"
ocupacaoAccountants = dados["OCCUPATION_TYPE"] == "Accountants"
ocupacaoMedicineStaff = dados["OCCUPATION_TYPE"] == "Medicine staff"
ocupacaoCooking = dados["OCCUPATION_TYPE"] == "Cooking staff"
ocupacaoSecurity = dados["OCCUPATION_TYPE"] == "Security staff"
ocupacaoCleaning = dados["OCCUPATION_TYPE"] == "Cleaning staff"
ocupacaoPrivateService = dados["OCCUPATION_TYPE"] == "Private service staff"
ocupacaoLowSkillLaborers = dados["OCCUPATION_TYPE"] == "Low-skill Laborers"
ocupacaoWaiters = dados["OCCUPATION_TYPE"] == "Waiters/barmen staff"
ocupacaoSecretaries = dados["OCCUPATION_TYPE"] == "Secretaries"
ocupacaoHR = dados["OCCUPATION_TYPE"] == "HR"
ocupacaoIT = dados["OCCUPATION_TYPE"] == "IT"
ocupacaoRealState = dados["OCCUPATION_TYPE"] == "Realty agents"