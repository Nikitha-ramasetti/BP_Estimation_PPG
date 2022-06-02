# reference and estimated values plot


#plot HR labels
plt.plot(df_result1.value[0:500], label='Reference HR', color= 'green')
plt.plot(df_result1.prediction[0:500], label='Estimated HR', color = 'red')
plt.xlabel("Time (windows)")
plt.ylabel("HR (mmHg)")
plt.legend()
plt.show()



#plot SBP labels
plt.figure(figsize=(12,4))
plt.plot(df_result2.value[0:500], label='Reference SBP', color='green')
plt.plot(df_result2.prediction[0:500], label='Estimated SBP', color='blue')
plt.xlabel("Time (windows)")
plt.ylabel("SBP (mmHg)")
plt.legend()
plt.show()


#plot DBP labels
plt.figure(figsize=(12,4))
plt.plot(df_result3.value, label='Reference DBP', color='green')
plt.plot(df_result3.prediction, label='Estimated DBP', color='orange')
plt.xlabel("Time (windows)")
plt.ylabel("DBP (mmHg)")
plt.legend()
plt.show()