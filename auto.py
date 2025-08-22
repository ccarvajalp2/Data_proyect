import streamlit as st
import pandas as pd
from datetime import date


col1, col2 = st.columns([3,1])

with col1:
    st.title("Automatización reporte Banca")

with col2:
    st.image(
        "https://www.bancodealimentos.org.ar/wp-content/uploads/2022/10/LOGO-BBVA-coreblue_RGB_DDB-1536x533.png", 
        width=250
    )


st.subheader("1. Selecciona el rango de fechas:")
min_fecha = date(2025, 5, 1)
max_fecha = date(2025, 5, 31)

fecha_inicio = st.date_input(
    "📅 Fecha de inicio",
    value=min_fecha,
    min_value=min_fecha,
    max_value=max_fecha
)

fecha_fin = st.date_input(
    "📅 Fecha final",
    value=max_fecha,
    min_value=min_fecha,
    max_value=max_fecha
)

st.write("Fecha inicio:", fecha_inicio)
st.write("Fecha fin:", fecha_fin)

st.subheader("2. Selecciona un agrupador:")
agrupadores = ['Seleccionar', 'Sucursal', 'Zona', 'Area', 'Segmento', 'Linea', 'Producto', 'Canal']
agrupador = st.selectbox("🔹 Selecciona un agrupador", agrupadores)

st.subheader("3. Selecciona los filtros que deseas realizar:")

filtros_dict = {
    "Sucursal": ['ACACIAS', 'ACOPI', 'AER CONNECTA 26', 'AGUACHICA', 'AGUAZUL', 'ALBANIA', 'ALCAZARES', 'ALTO PRADO', 'APARTADO', 'ARAUCA', 'ARMENIA CENTRO', 'ARMENIA NORTE', 'AVENIDA CERO', 'AVENIDA CHILE', 'AVENIDA EL DORADO', 'AVENIDA JIMENEZ', 'AVENIDA PANAMERICANA', 'AVENIDA ROOSEVELT', 'AVENIDA TERCERA NTE', 'BAHIA', 'BANCA WEALTH CAL 123', 'BARBOSA', 'BARRANCABERMEJA', 'BARRANCAS', 'BARRIO RESTREPO', 'BBVA CAJICA', 'BBVA CTO CIAL MALL P', 'BBVA SIEMENS', 'BBVA TOLEMAIDA', 'BBVA UNICENT PALMIR', 'BBVA WEALTH MAN PRIN', 'BCA WEALTH BMANGA', 'BCA WEALTH BOGOTA', 'BCA WEALTH BQUILLA', 'BCA WEALTH CALI', 'BCA WEALTH CALI SUR	', 'BCA WEALTH CARTAGENA', 'BCA WEALTH EL DORADO', 'BCA WEALTH EL NOGAL', 'BCA WEALTH MANIZALES', 'BCA WEALTH MEDELLIN', 'BCA WEALTH PEREIRA', 'BCA WEALTH WORLD TC', 'BELEN', 'BELLO', 'BELMIRA', 'BOCAGRANDE', 'BUCARAMANGA', 'BUENAVENTURA', 'BUGA', 'C CIAL UNICO VILLAVO', 'C.CALIMA ARMENIA', 'C.CIAL AV CHILE', 'CA#AVERAL', 'CABECERA DEL LLANO', 'CABLE PLAZA', 'CAFAM FLORESTA', 'CALARCA', 'CALASANZ', 'CALI', 'CALLE 80', 'CALLE 84', 'CALLE 97', 'CALLE DEL COMERCIO', 'CALLE NOVENA', 'CAN', 'CANTï¿½N NORTE', 'CARRERA 27', 'CARRERA 70', 'CARRERA OCHENTA', 'CARRERA ONCE', 'CARRERA PRIMERA', 'CARTAGENA', 'CARTAGO', 'CAUCASIA', 'CC BC WEALTH MANAGEM', 'CC BUENAVISTA', 'CC CEN FORMALIZACION', 'CC CHIPICHAPE', 'CC ENLACE APLICATIVO', 'CC FOGAFIN', 'CC PLATINO', 'CC UNICO - CALI', 'CC UNICO BQUILLA', 'CCIAL DIVER PLAZA', 'CCIAL MAYALES', 'CEDRITOS', 'CENABASTOS', 'CENTENARIO', 'CENTRAL DE ABASTOS', 'CENTRO 93', 'CENTRO ANDINO', 'CENTRO CHIA', 'CENTRO CIAL GRAN PLA', 'CENTRO CORPORATIVO', 'CENTRO MAYOR', 'CENTRO SUBA', 'CHAPINERO', 'CHIA', 'CHINU', 'CHIQUINQUIRA', 'CIMITARRA', 'CIUDAD DEL RIO', 'CIUDAD JARDIN', 'CIUDAD KENNEDY', 'CIUDAD SALITRE', 'COH  BUCARAMANGA', 'COH BARRANQUILLA', 'COH BOGOTA', 'COH CALI', 'COH MEDELLIN', 'COH PEREIRA', 'COLINA CAMPESTRE', 'COLSEGUROS', 'COLTEJER', 'COMERCIAL NORTE', 'CONGRESO DE LA REP', 'CONTADOR', 'CORABASTOS', 'CORFERIAS', 'CORPORATIVA CALI', 'CORPORATIVA MEDELLIN', 'COSMOCENTRO', 'COTA', 'COUNTRY', 'CRï¿½DITO CONSTRUCTOR', 'CT EMP OLAYA HERRERA', 'CTO CIAL UNICO PASTO', 'CTO INTERNACIONAL', 'CTRO CCIAL PRIMAVERA', 'CTRO EMPR CALLE 127', 'CUCUTA', 'CURUMANI', 'DANN', 'DEP DINERO ELECTRONC', 'DOS QUEBRADAS', 'DUITAMA', 'E BARRANQUILLA', 'E BUCARAMANGA', 'E CALI', 'E CARTAGENA', 'E CUCUTA', 'E IBAGUE', 'E MANIZALES', 'E NEIVA', 'E PEREIRA', 'E SANTA MARTA', 'E VILLAVICENCIO', 'EL BANCO', 'EL PARQUE', 'EL POBLADO', 'EL RECREO', 'EL TESORO', 'EMPRESAS SPARK', 'ENVIGADO', 'ESPINAL', 'EXITO', 'EXT OFICINA LA UNIï¿½N', 'EXT OFICINA SANTA FE', 'FACATATIVA', 'FADEGAN AUTOPIS NORT', 'FLORENCIA', 'FLORIDA', 'FONSECA', 'FONTIBON', 'FORTALEZA', 'FUNDACION', 'FUNZA', 'FUSAGASUGA', 'GALERIAS', 'GARZON', 'GENTE BBVA', 'GIRARDOT', 'GRAN BOULEVARD', 'GRAN VIA', 'GRANADA', 'GRANCENTRO', 'GREEN TOWERS', 'HACIENDA STA BARBARA', 'HAYUELOS', 'IMBANACO', 'INSTIT BMANGA', 'INSTITUCIONES CENTRO', 'IPIALES', 'ITAGUI', 'JAMUNDI', 'JARDIN PLAZA', 'KENNEDY CENTRAL', 'LA ALPUJARRA', 'LA CASTELLANA', 'LA CEJA', 'LA DORADA', 'LA ESTRADA', 'LA LOMA', 'LA PLAZUELA', 'LA TRIADA', 'LA TRINIDAD', 'LARANDIA', 'LAS AGUAS', 'LAURELES', 'LETICIA', 'LORICA', 'LOS ALMENDROS', 'LOS FUNDADORES', 'LOS MOLINOS', 'LP CORP BOGOTA', 'LP CORP CALI', 'MADRID', 'MAGANGUE', 'MAICAO', 'MALL LA FRONTERA', 'MANGA - CARTAGENA', 'MANIZALES', 'MARIQUITA', 'MEDELLIN', 'MELGAR', 'METROPOLITANO', 'MITï¿½', 'MOCOA', 'MODELIA', 'MOMPOS', 'MONTELIBANO', 'MONTERIA', 'MOSQUERA', 'MURILLO', 'NEIVA', 'NIZA', 'NORMANDIA', 'OCA#A', 'OCCIDENTE', 'OESTE CALI', 'OF AD T BTA CENTRAL', 'ORITO', 'OVIEDO', 'PAIPA', 'PALMIRA', 'PALOQUEMAO', 'PAMPLONA', 'PARALELO 108', 'PARQUE MURILLO', 'PARQUE NACIONAL', 'PASEO BOLIVAR', 'PASOANCHO', 'PASTO', 'PAZ DE ARIPORO', 'PEPE SIERRA', 'PEREIRA', 'PIEDECUESTA', 'PINARES', 'PITALITO', 'PLANETA RICA', 'PLATO', 'PLAZA 67', 'PLAZA IMPERIAL', 'PLAZA LAS AMERICAS', 'PLAZA LOPERENA', 'POPAYAN', 'PREMIUM PLAZA', 'PRIMERO DE MAYO', 'PRINCIPAL', 'PUENTE ARANDA', 'PUENTE LARGO', 'PUERTO ASIS', 'PUERTO BERRIO', 'PUERTO BOYACA', 'PUERTO CARRE#O', 'PUERTO GAITAN', 'PUERTO INIRIDA', 'PUERTO LOPEZ', 'QUIBDO', 'QUIRIGUA', 'REAL DE MINAS', 'REG EMP E INST ANT', 'REG EMP E INST NORTE', 'REG EMP E INST OCC', 'REGIONAL EMP BOGOTï¿½', 'REGIONAL GOB BOGOTï¿½', 'RIO PARQUE COMERCIAL', 'RIOHACHA', 'RIONEGRO', 'SABANALARGA', 'SABANETA', 'SAHAGUN', 'SALITRE PLAZA', 'SAN ANDRES', 'SAN GIL', 'SAN JOSï¿½ DE GUAVIARE', 'SAN MARCOS', 'SAN SIMON', 'SANCANCIO', 'SANTA FE MEDELLï¿½N', 'SANTA MARTA', 'SANTA MONICA', 'SANTA PAULA', 'SANTA ROSA DE CABAL', 'SANTAFE', 'SANTANDER DE QUILICH', 'SARAVENA', 'SAVANNA', 'SCHLUMBERGER', 'SEXTA AVENIDA', 'SIETE DE AGOSTO', 'SINCELEJO', 'SMART OFFICE', 'SOCORRO', 'SOGAMOSO', 'SOLEDAD', 'SUB R EMP E INST ANT', 'SUPEREJECUTIVOS', 'TAURAMENA', 'TELEFï¿½NICA', 'TOBERIN', 'TOCANCIPA', 'TULUA', 'TUNAL', 'TUNJA', 'TUNJA AV NORTE', 'TUQUERRES', 'TURBO', 'UBATE', 'UNICENTRO', 'UNICENTRO CALI', 'UNICENTRO MEDELLIN', 'UNICENTRO OCCIDENTE', 'UNICENTRO PASTO', 'UNICENTRO PEREIRA', 'URRAO', 'VALLEDUPAR', 'VENECIA', 'VILLAGARZON', 'VILLANUEVA', 'VILLAVICENCIO', 'VILLETA', 'WORLD TRADE', 'YOPAL', 'YUMBO', 'ZIPAQUIRA CENTRO'],
    "Zona": ['Z NB BAD BANK' ,'Z CARIBE NORTE' ,'Z CALI' ,'Z BOGOTï¿½ SUR' ,'Z ANTIOQUIA' ,'Z EJE CAFETERO' ,'Z CENTRO OCCIDENTE' ,'Z OCCIDENTE SUR' ,'Z SANTANDERES' ,'Z CC TYO OPERACIONES' ,'Z BC WEALTH BOGOTA' ,'Z CARIBE SUR' ,'Z MEDELLIN' ,'Z BOGOTï¿½ CENTRAL' ,'Z BARRANQUILLA' ,'Z CENTRO ORIENTE' ,'Z BOGOTA NORTE' ,'Z EMP Y GOB ANTIOQ' ,'Z ADMIN COM RESTO' ,'Z CC OPERACIONES' ,'Z  BARRANQUILLA' ,'Z PYME BTA CENTRAL' ,'Z MEDELLï¿½N' ,'Z PYME CARIBE NORTE' ,'Z BC WEALTH CENTRO' ,'Z EMPRESAS NORTE' ,'Z PYME OCCIDENTE SUR' ,'Z PYME SANTANDERES' ,'Z BC WEALTH NORTE' ,'Z CP CORPORATIVA' ,'Z PYME CENTRO OCC' ,'Z PYME CTO ORIENTE' ,'Z PYME CARIBE SUR' ,'Z G BE CENTRO' ,'Z BC WEALTH ANTIOQUI' ,'Z BC WEALTH OCCIDENT' ,'Z REGIONAL EMP BTA' ,'Z PYME BOGOTï¿½ SUR' ,'Z EMP Y GOB NORTE' ,'Z EMP Y GOB CENTRO' ,'Z E OCCIDENTE' ,'Z PYME CALI' ,'Z PYME BOGOTï¿½ NORTE' ,'Z PYME ANTIOQUIA' ,'Z LP CORPORATIVA' ,'Z SUBG EMP & GOB ANT' ,'Z BC DIR NAL PRIV BW' ,'Z REGIONAL GOB BTA' ,'Z CC OPERATIVIZACIï¿½N' ,'R EMP Y GOB OCCIDE' ,'Z EMPRESAS SPARK'],
    "Area": ['BANCA COMERCIAL', 'BANCA PYMES', 'BANCA WEALTH MANAGE', 'BANCA DE EMPRESAS', 'AREA CIB COLOMBIA', 'BANCA DE GOBIERNOS'],
    "Segmento": ['84000' ,'83000' ,'85100' ,'82000' ,'50100' ,'82100' ,'50300' ,'80000' ,'40100' ,'40300' ,'10100' ,'30100' ,'20100' ,'81000' ,'30200' ,'50200' ,'10200' ,'40200' ,'87100' ,'80100' ,'20600' ,'88000' ,'20000' ,'90500' ,'20200'],
    "Linea": ['CONSUMO TOTAL' ,'TOTAL EMPRESAS' ,'HIPOTECARIO' ,'ADMINISTRACION PUBLICA'],
    "Producto": ['CONSUMO LIBRE', 'LIBRANZAS', 'CUPO ROTATIVO', 'RESTO COMERCIAL', 'VEHICULOS', 'VIVIENDA', 'COMEX', 'REDESCUENTOS', 'FACTORING CONFIRMING Y TRIANGULAR', 'ADMINISTRACION PUBLICA', 'LEASING', 'AGROINDUSTRIA'],
    "Canal": ['DIGITAL', 'FVE', 'RED', 'CALL']
}

if agrupador == "Seleccionar":
    filtro_seleccionado = ["Todos"]
    st.info("No hay filtros disponibles")
else:
    opciones = filtros_dict.get(agrupador, [])
    filtro_seleccionado = st.multiselect(
        f"Selecciona l@s {agrupador}(s)(es) que deseas incluir", 
        opciones, 
        default=opciones
    )


st.subheader("4. Selecciona el formato de salida de la información:")
col_format1, col_format2, col_format3 = st.columns(3)
with col_format1:
    formato_html = st.checkbox("📄 HTML", value=False)
with col_format2:
    formato_pdf = st.checkbox("📝 PDF", value=False)
with col_format3:
    formato_csv = st.checkbox("📄 CSV", value=True)

if st.button("🚀 Generar informe"):
    with st.spinner("El proceso está en ejecución..."):
        from datetime import datetime
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import matplotlib.ticker as mtick

        fecha = datetime.today()

        #Importación de datos de información
        datos = pd.read_csv('https://raw.githubusercontent.com/ccarvajalp2/Data_proyect/refs/heads/main/Banca%20prueba.csv', sep = ';')

        datos['cust_consm_loan_cr_grant_amount'] = datos['cust_consm_loan_cr_grant_amount'] / 1000
        datos['last_change_contract_date'] = pd.to_datetime(datos['last_change_contract_date'], dayfirst=True)
        datos['birth_date'] = pd.to_datetime(datos['birth_date'], dayfirst=True)
        moda = datos['customer_antiquity_date'].mode()[0]
        datos['customer_antiquity_date'] = datos['customer_antiquity_date'].apply(lambda x: moda if '/' not in x else x)
        datos['customer_antiquity_date'] = pd.to_datetime(datos['customer_antiquity_date'], dayfirst=True)

        #Transformación de datos
        edad = fecha - datos['birth_date']
        datos['edad'] = edad.dt.days / 365
        antiguedad = fecha - datos['customer_antiquity_date']
        datos['antiguedad'] = antiguedad.dt.days / 365
        
        # Asignación de metas
        metas = datos.groupby(['branch_id', 'product_name']).agg(Meta=('cust_consm_loan_cr_grant_amount', 'sum')).reset_index()
        metas = metas.groupby('product_name').agg(Meta=('Meta', 'mean')).reset_index()

        diccionario = pd.DataFrame({
            "clave": ["Sucursal", "Zona", "Area", 'Segmento', 'Linea', 'Producto', 'Canal'],
            "valor": ["branch_name", "zone_branch_name", "business_area_name", 'segment_type', 'product_line_name', 'product_name', 'sale_end_channel_name']
        })

        agrupador = diccionario.loc[diccionario["clave"] == agrupador, "valor"].values[0]

        fecha_inicio = pd.to_datetime(fecha_inicio)
        fecha_fin = pd.to_datetime(fecha_fin)
        filtro = datos[(datos['last_change_contract_date'] >= fecha_inicio) & (datos['last_change_contract_date'] <= fecha_fin)]
        filtro = filtro[filtro[agrupador].isin(filtro_seleccionado)]

        #Datos agrupados
        data = filtro.groupby(['branch_id', 'branch_name', 'zone_branch_name', 'business_area_name', 'segment_type', 'product_line_name', 'product_name', 'sale_end_channel_name']).agg(Transacciones=('branch_id', 'count'), Monto=('cust_consm_loan_cr_grant_amount', 'sum'), Edad=('edad', 'mean'), Antiguedad=('antiguedad', 'mean')).reset_index()
        data = pd.merge(data, metas, on = 'product_name')
        
        #Tendencias
        tendencia = filtro.groupby(['last_change_contract_date', 'branch_id', 'branch_name', 'zone_branch_name', 'business_area_name', 'segment_type', 'product_line_name', 'product_name', 'sale_end_channel_name']).agg(Monto=('cust_consm_loan_cr_grant_amount', 'sum')).reset_index()

        #Generación de la información

        agrupado = data.groupby(agrupador).agg(Colocaciones= ('Transacciones', 'sum'), Monto=('Monto', 'sum'), Meta=('Meta', 'sum'), Edad=('Edad', 'mean'), Antiguedad=('Antiguedad', 'mean')).round(0).reset_index()
        agrupado['Cumplimiento'] = agrupado['Monto'] / agrupado['Meta']
        agrupado['Cumple'] = np.where(agrupado['Cumplimiento'] >= 1, 'Si', 'No')
        agrupado = agrupado.sort_values('Cumplimiento', ascending=False)
        agrupado['Cumplimiento'] = agrupado['Cumplimiento'].clip(upper=1)
        agrupado['GAP'] = 1- agrupado['Cumplimiento']
        agrupado = agrupado.reset_index(drop=True)
        agrupado.insert(0, 'Posición', range(1, len(agrupado)+1))

        meta = agrupado['Meta'].sum()
        meta = f"${meta:,.0f}"

        #Tendencia

        tend = tendencia.groupby(['last_change_contract_date']).agg(Monto=('Monto', 'sum')).reset_index()
        tend = tend.sort_values('last_change_contract_date')
        tend['Cumplimiento acumulado'] = tend['Monto'].cumsum()
        tend['Meta'] = agrupado['Meta'].sum()
        tend['Cumplimiento acumulado'] = tend['Cumplimiento acumulado']/1000000
        tend['Meta'] = tend['Meta']/1000000

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(tend['last_change_contract_date'].dt.day.astype(str), tend['Cumplimiento acumulado'], color= "#004481", label='Cumplimiento acumulado', alpha=0.9)
        ax.plot(tend['last_change_contract_date'].dt.day.astype(str), tend['Meta'], color='#BFC0C0', marker="o", label="Meta")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}M'))
        ax.set_xlabel("Dia del mes")
        ax.set_ylabel("Colocaciones")
        ax.set_title("Montos otorgados acumulados vs meta")
        ax.legend()
        plt.show()


        # Gráfico de dispersión
        plt.figure(figsize=(7,5))
        sns.scatterplot(
            data=agrupado,
            x="Edad", 
            y="Antiguedad",
            hue="Cumple",        
            palette=['#004481', '#BFC0C0'],    
            s=100               
        )
        plt.title("Edad promedio vs antigüedad promedio de clientes \npor cumplimiento de sucursales")
        plt.xlabel("Edad promedio")
        plt.ylabel("Antigüedad promedio")
        plt.show()



        plt.figure(figsize=(7,5))
        sns.scatterplot(
            data=agrupado,
            x="Colocaciones", 
            y="Monto",
            hue="Cumple",       
            palette=['#004481', '#BFC0C0'],   
            s=100               
        )
        plt.title("Colocaciones vs montos otorgados \npor cumplimiento de sucursales")
        plt.xlabel("Colocaciones")
        plt.ylabel("Montos")
        plt.show()


        #Grafico de Dona
        Cumplimiento = agrupado['Cumplimiento'].mean()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            [Cumplimiento, 1-Cumplimiento],
            colors=["#004481", "#BFC0C0"], 
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3) 
        )
        ax.text(0, 0, f"{Cumplimiento*100:.2f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#004481")
        plt.title("Cumplimiento")
        plt.show()

        GAP = agrupado['GAP'].mean()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            [GAP, 1-GAP],
            colors=["#004481", "#BFC0C0"], 
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3) 
        )
        ax.text(0, 0, f"{GAP*100:.2f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#004481")
        plt.title("GAP")
        plt.show()
        
        agrupado2 = agrupado.drop(columns=['Cumple'])
        agrupado2["Monto"] = agrupado2["Monto"]/1000000
        agrupado2["Meta"]  = agrupado2["Meta"]/1000000
        agrupado2["Monto"] = agrupado2["Monto"].apply(lambda x: f"${x:,.0f}")
        agrupado2["Meta"]  = agrupado2["Meta"].apply(lambda x: f"${x:,.0f}")
        agrupado2["Cumplimiento"] = agrupado2["Cumplimiento"].apply(lambda x: f"{x:.2%}")
        agrupado2["GAP"] = agrupado2["GAP"].apply(lambda x: f"{x:.1%}")

        agrupado3 = agrupado.drop(columns=['Cumple'])
        agrupado3 = agrupado3[(agrupado3['Colocaciones'] >= 100) & (agrupado3['Colocaciones'] <= 1000)]
        agrupado3 = agrupado3.sort_values('GAP', ascending=False)
        agrupado3 = agrupado3[agrupado3['GAP']>=.6]
        agrupado3["Monto"] = agrupado3["Monto"]/1000000
        agrupado3["Meta"]  = agrupado3["Meta"]/1000000
        agrupado3["Monto"] = agrupado3["Monto"].apply(lambda x: f"${x:,.0f}")
        agrupado3["Meta"]  = agrupado3["Meta"].apply(lambda x: f"${x:,.0f}")
        agrupado3["Cumplimiento"] = agrupado3["Cumplimiento"].apply(lambda x: f"{x:.2%}")
        agrupado3["GAP"] = agrupado3["GAP"].apply(lambda x: f"{x:.1%}")


        plt.close("all")


        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(tend['last_change_contract_date'].dt.day.astype(str), tend['Cumplimiento acumulado'], color= "#004481", label='Cumplimiento acumulado', alpha=0.9)
        ax.plot(tend['last_change_contract_date'].dt.day.astype(str), tend['Meta'], color='#BFC0C0', marker="o", label="Meta")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}M'))
        ax.set_xlabel("Dia del mes")
        ax.set_ylabel("Colocaciones")
        ax.set_title("Montos otorgados acumulados vs meta")
        ax.legend()
        fig.savefig("grafico_tendencia.png", bbox_inches="tight")


        Cumplimiento = agrupado['Cumplimiento'].mean()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie([Cumplimiento, 1-Cumplimiento], colors=["#004481", "#BFC0C0"], startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{Cumplimiento*100:.2f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#004481")
        plt.title("Cumplimiento")
        fig.savefig("grafico_cumplimiento.png", bbox_inches="tight")


        GAP = agrupado['GAP'].mean()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie([GAP, 1-GAP], colors=["#004481", "#BFC0C0"], startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{GAP*100:.2f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#004481")
        plt.title("GAP")
        fig.savefig("grafico_gap.png", bbox_inches="tight")

        from io import BytesIO
        import base64

        from io import BytesIO
        import zipfile

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:


            if formato_html:
                enumerador = 1
                html = """
                <html>
                <head><meta charset="UTF-8"></head>
                <body>
                <img src="https://www.bancodealimentos.org.ar/wp-content/uploads/2022/10/LOGO-BBVA-coreblue_RGB_DDB-1536x533.png" width="100"><br><br>
                <h1 style="color:#004481;">Reporte de Banca</h1>
                """
                html += f"<p>Actualización: {fecha}</p>"
                html += agrupado2.to_html(index=False)
                html += "</body></html>"

                zf.writestr("Reporte_Banca_BBVA.html", html)


            if formato_pdf:
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import letter
                from reportlab.lib import colors

                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("Reporte de Banca", styles['Title']))
                story.append(Spacer(1,12))
                story.append(Paragraph(f"Actualización: {fecha}", styles['Normal']))



                doc.build(story)
                pdf_buffer.seek(0)
                zf.writestr("Reporte_Banca_BBVA.pdf", pdf_buffer.read())


            if formato_csv:
                csv_content = agrupado.to_csv(index=False)
                zf.writestr("Reporte_Banca_BBVA.csv", csv_content)


        zip_buffer.seek(0)


        st.download_button(
            label="📥 Descargar Reportes Seleccionados (ZIP)",
            data=zip_buffer,
            file_name="Reportes_Banca_BBVA.zip",
            mime="application/zip"
        )
        st.success("✅ Proceso finalizado. Haz clic en el botón de descarga.")
