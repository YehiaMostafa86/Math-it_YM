from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    path('plot',views.graph,name='plot'),
    path('update_graph', views.update_graph, name='update_graph'),
    path ('send_gemeni',views.send_gemeni,name='send_gemeni'), 
    path ('send_deepseek',views.send_deepseek,name='send_deepseek'),
    path ('send_welfram',views.send_welfram,name='send_welfram'),
    path ('send_llama',views.send_llama,name='send_llama'),
    path ('send_sympolab',views.send_sympolab,name='send_sympolab'),
    path ('send_stackexchange',views.send_stackexchange,name='send_stackexchange'),
]