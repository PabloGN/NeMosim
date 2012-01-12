INSTALL(FILES
		Input.ini
		PoissonSource.ini
		Izhikevich.ini
		IzhikevichRS.ini
		Kuramoto.ini
	DESTINATION ${NEMO_SYSTEM_PLUGIN_DIR})

# Plugins which rely on separate excitatory and inhibitory currents are only
# valid if we do indeed keep these separate.
IF(NOT NEMO_SINGLE_CURRENT)
	INSTALL(FILES
		IF_curr_exp.ini
	DESTINATION ${NEMO_SYSTEM_PLUGIN_DIR})
ENDIF(NOT NEMO_SINGLE_CURRENT)
