#include "dyn_load.hpp"

#ifdef _MSC_VER

#error "Windows loading not yet implemented"

#else

bool
dl_init()
{
	return lt_dlinit() == 0;
}

bool
dl_exit()
{
	return lt_dlexit() == 0;
}

dl_handle
dl_load(const char* name)
{	
	return lt_dlopenext(name);
}

bool
dl_unload(dl_handle h)
{
	return lt_dlclose(h) == 0;
}

const char*
dl_error()
{
	return lt_dlerror();
}

void*
dl_sym(dl_handle hdl, const char* name)
{
	return lt_dlsym(hdl, name);
}

#endif