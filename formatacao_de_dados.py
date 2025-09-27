import locale
from datetime import datetime

from jinja2 import FileSystemLoader, Environment

from processamento_de_dados import carregar_tabelas

def pegar_template_renderizado(
        mes_referencia,
        pasta_dados,
        arquivo_excel,
        pasta_assets,
        arquivo_template,
        arquivo_css
):
    dict_tabelas = carregar_tabelas(
        mes_referencia=mes_referencia,
        pasta_dados=pasta_dados,
        arquivo_excel=arquivo_excel,
    )
    template = carregar_template(pasta_assets, arquivo_template)

    caminho_css = pasta_assets / arquivo_css
    css = carregar_css(caminho_css)

    return renderizar_template_como_html(
        template=template,
        css=css,
        mes_referencia=mes_referencia,
        dict_tabelas=dict_tabelas,
    )

def carregar_template(
        pasta_assets,
        arquivo_template,
):
    loader = FileSystemLoader(pasta_assets)
    environment = Environment(loader=loader)
    template = environment.get_template(arquivo_template)
    return template

def carregar_css(caminho_css):
    with open(caminho_css) as arquivo:
        css = arquivo.read()
    return css 

def renderizar_template_como_html(
        template,
        css,
        mes_referencia,
        dict_tabelas,
):
    agora = datetime.now()
    dia = agora.strftime('%d/%m/%Y')
    hora = agora.strftime('%H:%M')

    template_vars = {
        'stylesheet': css,
        'mes_referencia': mes_referencia,
        'dia': dia,
        'hora': hora,
    }
    for nome_tabela, tabela in dict_tabelas.items():
        template_vars[nome_tabela] = tabela.to_html(float_format=formatar_valor)
    string_html = template.render(**template_vars)
    return string_html


def formatar_valor(valor):
    lingua = 'pt_BR.UTF-8'
    locale.setlocale(locale.LC_ALL, lingua)
    return locale.currency(valor, grouping=True)